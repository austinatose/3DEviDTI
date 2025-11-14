import os
import torch
import pandas as pd
import numpy as np
import warnings
from sklearn.metrics import matthews_corrcoef, confusion_matrix, f1_score, recall_score, precision_score, accuracy_score, roc_auc_score, precision_recall_curve, auc, roc_curve
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import matplotlib.pyplot as plt

from config.cfg import get_cfg_defaults
from dataset import MyDataset, collate_fn

from model import Model

def dirichlet_loss(predictions, labels, alpha=1.0):
    """
    Compute Dirichlet loss for multi-class classification.
    predictions: Tensor of shape (batch_size, num_classes)
    labels: Tensor of shape (batch_size,)
    alpha: concentration parameter
    """
    num_classes = predictions.size(1)
    one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()
    S = torch.sum(predictions, dim=1, keepdim=True)
    log_S = torch.log(S)
    log_p = torch.digamma(predictions) - torch.digamma(S)

    loss = torch.sum((one_hot_labels - (predictions / S)) * log_p, dim=1)
    return torch.mean(loss)

class Solver:
    def __init__(self, model, cfg, device, optim, loss_fn, eval):
        self.cfg = cfg
        self.device = device
        self.model = model.to(self.device)
        self.batch_size = cfg.SOLVER.BATCH_SIZE
        self.epochs = cfg.SOLVER.EPOCHS
        self.learning_rate = cfg.SOLVER.LR
        self.weight_decay = cfg.SOLVER.WEIGHT_DECAY
        self.optim = optim(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, mode='min', factor=0.5, patience=5)  # FIXME

        # use pre-split data first, then implement k-fold later
        self.train_ds = MyDataset(cfg.DATA.TRAIN_CSV_PATH, cfg.DATA.PROTEIN_DIR, cfg.DATA.DRUG_DIR)
        self.train_dl = DataLoader(self.train_ds, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate_fn, drop_last=True)
        
        self.test_ds = MyDataset(cfg.DATA.TEST_CSV_PATH, cfg.DATA.PROTEIN_DIR, cfg.DATA.DRUG_DIR)
        self.test_dl = DataLoader(self.test_ds, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate_fn, drop_last=False)
        self.val_ds = MyDataset(cfg.DATA.VAL_CSV_PATH, cfg.DATA.PROTEIN_DIR, cfg.DATA.DRUG_DIR)
        self.val_dl = DataLoader(self.val_ds, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate_fn, drop_last=False)

        self.loss_fn = dirichlet_loss

        print("Solver initialized.")

    def predict(self, data_loader, epoch, optim=None):
        running_loss = 0.0
        # TODO: running accuracy required? batch size being around 32 is too small to detect changes
        results = []
        for i, batch in enumerate(data_loader):
            labels = batch["label"].to(self.device)  # [batchsize]
            protein_mask = batch["protein_mask"].to(self.device)  # [batchsize, Lpmax]
            drug_mask = batch["drug_mask"].to(self.device)  # [batchsize, Ldmax]
            protein_emb = batch["protein_emb"].to(self.device)  # [batchsize, Lpmax, Dp]
            drug_emb = batch["drug_emb"].to(self.device)  # [batchsize, Ldmax, Dd]

            # padding is done
            # sequence_lengths = metadata['length'][:, None].to(self.device)  # [batchsize, 1]
            # max_len = sequence_lengths.max().item()

            predictions = self.model(protein_emb, drug_emb, protein_mask=protein_mask, drug_mask=drug_mask, mode="train")  # [batchsize, 2]

            # 0 is negative, 1 is positive. take larger logit as pred
            _, pred = torch.max(predictions, dim=1)  # [batchsize]

            loss = self.loss_fn(predictions, labels) # TODO: compute loss

            results.append(torch.stack([labels.detach().cpu(), pred.detach().cpu()], dim=1).numpy())  # [batchsize, 2]

            if optim:  # run backpropagation if an optimizer is provided
                loss.backward()
                self.optim.step()
                self.optim.zero_grad()

            running_loss += loss.item()

            if i % 3 == 0:  # log every log_iterations
                if epoch:
                    # get pred from alphas
                    print('Epoch %d ' % (epoch), end=' ')
                    print('[Iter %5d/%5d] %s: loss: %.7f, accuracy: %.4f%%' % (
                        i + 1, len(data_loader), 'Train' if optim else 'Val', loss.item(),
                        100 * (labels==pred).sum().item() / labels.size(0)))
        running_loss /= len(data_loader)
        return running_loss, np.concatenate(results, axis=0)  # [N, 2]
    
    @torch.no_grad()
    def predict_test(self, data_loader):  # evaluation-time, evidential outputs
        self.model.eval()
        pred_pairs = []     # [pred, label]
        prob_list = []      # P(y=1)
        conf_list = []      # confidence proxy: K / sum(alpha)
        ev_list = []        # evidence = alpha - 1
        bk_list = []        # belief = (alpha - 1) / sum(alpha)
        label_list = []
        uniprot_ids = []
        drugbank_ids = []

        for _, batch in enumerate(data_loader):
            labels = batch["label"].to(self.device)                    # (B,)
            protein_mask = batch["protein_mask"].to(self.device)       # (B, Lp)
            drug_mask = batch["drug_mask"].to(self.device)             # (B, Ld)
            protein_emb = batch["protein_emb"].to(self.device)         # (B, Lp, Dp)
            drug_emb = batch["drug_emb"].to(self.device)               # (B, Ld, Dd)

            # Model outputs Dirichlet alphas (>=1) of shape (B, 2)
            alphas = self.model(protein_emb, drug_emb,
                                 protein_mask=protein_mask,
                                 drug_mask=drug_mask,
                                 mode="test")                          # (B,2)

            # Move to CPU numpy for post-processing
            alphas_np = alphas.detach().cpu().numpy()                   # (B,2)
            labels_np = labels.detach().cpu().numpy().astype(int)       # (B,)

            # Per-batch computations
            sum_alpha = np.sum(alphas_np, axis=1, keepdims=True)        # (B,1)
            probs = alphas_np / sum_alpha                               # (B,2)
            preds = np.argmax(probs, axis=1).astype(int)                # (B,)

            # Confidence proxy (higher sum_alpha -> higher confidence)
            K = alphas_np.shape[1]
            conf = K / sum_alpha.squeeze(1)                              # (B,)

            evidence = alphas_np - 1.0                                   # (B,2)
            belief = evidence / sum_alpha                                # (B,2)

            pred_pairs.append(np.stack([preds, labels_np], axis=1))      # (B,2)
            prob_list.extend(probs[:, 1].tolist())                       # positive-class prob
            conf_list.extend(conf.tolist())
            for i in range(alphas_np.shape[0]):
                ev_list.append(evidence[i])
                bk_list.append(belief[i])

            uniprot_ids.extend(batch["uniprot_id"])                     # list[str]
            drugbank_ids.extend(batch["drugbank_id"])                           # list[str]

        pred_pairs = np.vstack(pred_pairs) if len(pred_pairs) else np.zeros((0,2), dtype=int)
        return uniprot_ids, drugbank_ids, pred_pairs, conf_list, conf_list, prob_list, ev_list, bk_list

    def train(self, train_loader, val_loader):
        no_improve_epochs = 0 # TODO
        for epoch in range(self.epochs):
            print(f"\n=== Epoch {epoch + 1}/{self.epochs} ===")
            self.model.train()
            train_loss, train_results = self.predict(train_loader, epoch + 1, optim=self.optim)
            self.model.eval()

            with torch.no_grad():
                _, _, val_results, _, _, prob_list, _, _ = self.predict_test(val_loader)

            val_results = np.squeeze(np.array(val_results))  # [N, 2]
            train_acc = 100 * np.equal(train_results[:, 0], train_results[:, 1]).sum() / len(train_results)
            val_acc = 100 * np.equal(val_results[:, 0], val_results[:, 1]).sum() / len(val_results)
            with warnings.catch_warnings():  # because sklearns mcc implementation is a little dim
                warnings.filterwarnings("ignore", message="invalid value encountered in double_scalars")
                train_mcc = matthews_corrcoef(train_results[:, 1], train_results[:, 0])
                val_mcc = matthews_corrcoef(val_results[:, 1], val_results[:, 0])
                val_auc = roc_auc_score(val_results[:, 1], prob_list)

            print('[Epoch %d] val accuracy: %.4f%% train accuracy: %.4f%%' % (epoch + 1, val_acc, train_acc))

        # TODO: whole bunch incomplete


    def evaluate(self, data_loader):
        pass

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
    # Additional methods for training, validation, testing would go here

if __name__ == "__main__":
    cfg = get_cfg_defaults()
    model = Model(cfg)
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    solver = Solver(model, cfg, device=device,
                    optim=torch.optim.Adam, loss_fn=None, eval=None)

    solver.train(solver.train_dl, solver.val_dl)