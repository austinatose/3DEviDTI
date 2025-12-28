import os
import torch
import pandas as pd
import numpy as np
import warnings
from sklearn.metrics import matthews_corrcoef, confusion_matrix, f1_score, recall_score, precision_score, accuracy_score, roc_auc_score, precision_recall_curve, auc, roc_curve, average_precision_score
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import matplotlib.pyplot as plt

from config.cfg import get_cfg_defaults
from dataset import MyDataset, collate_fn, KIBADataset

from model import Model

import torch
import torch.nn.functional as F

def dirichlet_kl_to_uniform(alpha):
    """
    KL(Dir(alpha) || Dir(1,...,1)) per sample.
    alpha: (B, K), all > 0
    returns: (B, 1)
    """
    beta = torch.ones_like(alpha)
    S_alpha = alpha.sum(dim=-1, keepdim=True)
    S_beta = beta.sum(dim=-1, keepdim=True)

    # log B(·)
    log_B_alpha = torch.lgamma(alpha).sum(dim=-1, keepdim=True) - torch.lgamma(S_alpha)
    log_B_beta  = torch.lgamma(beta).sum(dim=-1, keepdim=True) - torch.lgamma(S_beta)

    # KL = log B(beta) - log B(alpha) + sum((alpha-beta)(ψ(alpha) - ψ(S_alpha)))
    digamma_alpha = torch.digamma(alpha)
    digamma_S_alpha = torch.digamma(S_alpha)

    kl = (log_B_beta - log_B_alpha
          + ((alpha - beta) * (digamma_alpha - digamma_S_alpha)).sum(dim=-1, keepdim=True))
    return kl


def dirichlet_loss(alphas, y, lam=0.1):
    """
    Sensoy-style evidential loss:
      L = |y - p|^2 + p(1-p)/(S+1) + λ * KL(Dir(alpha_hat) || Dir(1))

    y:      (B,) int labels OR (B, K) one-hot labels
    alphas: (B, K) positive Dirichlet parameters (evidence + 1)
    lam:    coefficient for KL term
    """
    B, K = alphas.shape

    # Make y one-hot
    if y.dim() == 1:
        y_one_hot = F.one_hot(y.long(), num_classes=K).float()
    else:
        y_one_hot = y.float()

    y_one_hot = y_one_hot.to(alphas.device)

    # Dirichlet mean
    S = alphas.sum(dim=-1, keepdim=True)      # (B,1)
    p = alphas / S                            # (B,K)

    # Squared error + variance term
    A = ((y_one_hot - p) ** 2).sum(dim=-1, keepdim=True)
    B_term = (p * (1 - p) / (S + 1)).sum(dim=-1, keepdim=True)
    sos = A + B_term                          # (B,1)

    # Regularization KL term
    alpha_hat = y_one_hot + (1 - y_one_hot) * alphas
    # kl_reg = lam * dirichlet_kl_to_uniform(alpha_hat)  # (B,1)

    kl_reg = dirichlet_kl_to_uniform(alpha_hat)  # (B,1)
    error_weight = A.detach()    # or A.detach() / A.detach().mean().clamp_min(1e-6)
    # print("Error weight:", error_weight)
    reg = lam * error_weight * kl_reg
    # print("KL reg:", reg)

    loss = sos + reg                       # (B,1)
    return loss.mean()              # scalar

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
        self.train_ds = KIBADataset(cfg.DATA.TRAIN_CSV_PATH, cfg.DATA.PROTEIN_DIR, cfg.DATA.DRUG_DIR)
        self.train_dl = DataLoader(self.train_ds, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate_fn, drop_last=True)
        
        self.test_ds = KIBADataset(cfg.DATA.TEST_CSV_PATH, cfg.DATA.PROTEIN_DIR, cfg.DATA.DRUG_DIR)
        self.test_dl = DataLoader(self.test_ds, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate_fn, drop_last=False)
        self.val_ds = KIBADataset(cfg.DATA.VAL_CSV_PATH, cfg.DATA.PROTEIN_DIR, cfg.DATA.DRUG_DIR)
        self.val_dl = DataLoader(self.val_ds, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate_fn, drop_last=False)

        self.loss_fn = dirichlet_loss if loss_fn == "dirichlet_loss" else F.cross_entropy
        self.start_date = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

        self.max_val_acc = 0.0

        # hash together cfg and model and date to identify this training run
        self.id = hash((str(cfg), str(model), self.start_date))
        print(f"Training run ID: {self.id}")

        with open(os.path.join("logs", f"training_log_{self.start_date}.csv"), "w") as f:
            f.write(self.start_date + "\n")
            f.write(str(self.id) + "\n\n")
            f.write(str(cfg))
            # write model components
            f.write("\n\nModel architecture:\n")
            f.write(str(model))
            f.write("\n")

        print(f"Using loss function: {self.loss_fn.__name__}")

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

            # old_params = []
            # for p in model.parameters():
            #     old_params.append(p.detach().clone())

            # padding is done
            # sequence_lengths = metadata['length'][:, None].to(self.device)  # [batchsize, 1]
            # max_len = sequence_lengths.max().item()

            predictions = self.model(protein_emb, drug_emb, protein_mask=protein_mask, drug_mask=drug_mask, mode="train")  # [batchsize, 2]
            # print("Predictions:", predictions)
            # print("Labels:", labels)

            # 0 is negative, 1 is positive. take larger logit as pred
            _, pred = torch.max(predictions, dim=1)  # [batchsize]
            # print("logits:", predictions[:5])
            # print("preds:", predictions[:5].argmax(dim=1))
            # print("labels:", labels[:5])

            # # if all predictions are the same, something's wrong
            # if (pred == pred[0]).all():
            #     print("爆炸了: all predictions are the same in this batch")

            loss = self.loss_fn(predictions, labels) 
            # loss = F.cross_entropy(predictions, labels)

            results.append(torch.stack([labels.detach().cpu(), pred.detach().cpu()], dim=1).numpy())  # [batchsize, 2]

            if optim:  # run backpropagation if an optimizer is provided
                loss.backward()
                # for name, p in model.named_parameters():
                #     if p.grad is not None:
                #         print("grad norm", name, p.grad.norm().item())
                #         break
                self.optim.step()
                optim.zero_grad()

            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         grad_mean = param.grad.abs().mean()
            #         if grad_mean == 0:
            #             print(f"Zero gradients in {name}")
            #         elif torch.isnan(param.grad).any():
            #             print(f"NaN gradients in {name}")
            
            # diff = 0.0
            # for p, old in zip(model.parameters(), old_params):
            #     diff += (p - old).abs().sum().item()
            # print("Total parameter change:", diff)

            running_loss += loss.item()

            if i % 100 == 0:  # log every log_iterations
                if epoch:
                    # get pred from alphas
                    print('Epoch %d ' % (epoch), end=' ')
                    print('[Iter %5d/%5d] %s: loss: %.7f, accuracy: %.4f%%' % (
                        i + 1, len(data_loader), 'Train' if optim else 'Val', loss.item(),
                        100 * (labels==pred).sum().item() / labels.size(0)))
        running_loss /= len(data_loader)
        return running_loss, np.concatenate(results, axis=0)  # [N, 2]
    
    def predict_test(self, data_loader):  # evaluation-time, evidential outputs
        self.model.eval()
        pred_pairs = []     # [pred, label]
        prob_list = []      # P(y=1)
        conf_list = []      # confidence proxy: K / sum(alpha) or max prob
        ev_list = []        # evidence = alpha - 1
        bk_list = []        # belief = (alpha - 1) / sum(alpha)
        label_list = []
        uniprot_ids = []
        drug_ids = []

        for _, batch in enumerate(data_loader):
            labels = batch["label"].to(self.device)                    # (B,)
            protein_mask = batch["protein_mask"].to(self.device)       # (B, Lp)
            drug_mask = batch["drug_mask"].to(self.device)             # (B, Ld)
            protein_emb = batch["protein_emb"].to(self.device)         # (B, Lp, Dp)
            drug_emb = batch["drug_emb"].to(self.device)               # (B, Ld, Dd)

            labels_np = labels.detach().cpu().numpy().astype(int)       # (B,)

            if self.loss_fn is F.cross_entropy:
                # CE mode: model outputs logits; use softmax for probabilities
                logits = self.model(protein_emb, drug_emb,
                                    protein_mask=protein_mask,
                                    drug_mask=drug_mask,
                                    mode="test")                       # (B,2)

                probs_t = F.softmax(logits, dim=1)                      # (B,2)
                probs = probs_t.detach().cpu().numpy()
                preds = np.argmax(probs, axis=1).astype(int)            # (B,)

                # Simple confidence proxy: max class probability
                conf = probs.max(axis=1)                                # (B,)

                # For CE mode, we don't have evidential quantities; append zeros
                evidence = np.zeros_like(probs)                         # (B,2)
                belief = np.zeros_like(probs)                           # (B,2)
            else:
                # Evidential mode: model outputs Dirichlet alphas (>=1) of shape (B, 2)
                alphas = self.model(protein_emb, drug_emb,
                                   protein_mask=protein_mask,
                                   drug_mask=drug_mask,
                                   mode="test")                          # (B,2)

                alphas_np = alphas.detach().cpu().numpy()               # (B,2)
                sum_alpha = np.sum(alphas_np, axis=1, keepdims=True)    # (B,1)
                probs = alphas_np / sum_alpha                           # (B,2)
                preds = np.argmax(probs, axis=1).astype(int)            # (B,)

                # Confidence proxy (higher sum_alpha -> higher confidence)
                K = alphas_np.shape[1]
                conf = (K / sum_alpha.squeeze(1))                       # (B,)

                evidence = alphas_np - 1.0                              # (B,2)
                belief = evidence / sum_alpha                           # (B,2)

            pred_pairs.append(np.stack([preds, labels_np], axis=1))     # (B,2)
            prob_list.extend(probs[:, 1].tolist())                      # positive-class prob
            conf_list.extend(conf.tolist())
            for i in range(probs.shape[0]):
                ev_list.append(evidence[i])
                bk_list.append(belief[i])

            uniprot_ids.extend(batch["uniprot_id"])                     # list[str]
            drug_ids.extend(batch["drug_id"])                   # list[str]

        pred_pairs = np.vstack(pred_pairs) if len(pred_pairs) else np.zeros((0,2), dtype=int)
        return uniprot_ids, drug_ids, pred_pairs, conf_list, conf_list, prob_list, ev_list, bk_list

    def train(self, train_loader, val_loader):
        no_improve_epochs = 0 # TODO
        best_val_acc = float('-inf')
        for epoch in range(self.epochs):
            print(f"\n=== Epoch {epoch + 1}/{self.epochs} ===")
            epoch_start_time = pd.Timestamp.now()
            self.model.train()
            train_loss, train_results = self.predict(train_loader, epoch + 1, optim=self.optim)
            self.model.eval()

            with torch.no_grad():
                _, _, val_results, _, _, prob_list, _, _ = self.predict_test(val_loader)

            cm = confusion_matrix(val_results[:, 1], val_results[:, 0])
            TN, FP, FN, TP = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

            val_results = np.squeeze(np.array(val_results))  # [N, 2]
            train_acc = 100 * np.equal(train_results[:, 0], train_results[:, 1]).sum() / len(train_results)
            val_acc = 100 * np.equal(val_results[:, 0], val_results[:, 1]).sum() / len(val_results)
            with warnings.catch_warnings():  # because sklearns mcc implementation is a little dim
                warnings.filterwarnings("ignore", message="invalid value encountered in double_scalars")
                train_mcc = matthews_corrcoef(train_results[:, 1], train_results[:, 0])
                val_mcc = matthews_corrcoef(val_results[:, 1], val_results[:, 0])
                val_auc = roc_auc_score(val_results[:, 1], prob_list)
                val_auprc = average_precision_score(val_results[:, 1], prob_list)

            print('[Epoch %d] val accuracy: %.4f%% train accuracy: %.4f%% train loss: %.4f val AUC: %.4f val AUPRC: %.4f' % (epoch + 1, val_acc, train_acc, train_loss, val_auc, val_auprc))

            self.save_model(epoch + 1) # remember to delete
            if val_acc > self.max_val_acc:
                self.max_val_acc = val_acc
                self.save_model(epoch + 1, dir="best_models")
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            # create new file with current date and time if not exist
            date = self.start_date
            epoch_time = (pd.Timestamp.now() - epoch_start_time).total_seconds()
            print(f"Epoch time: {epoch_time:.2f} seconds")
            if not os.path.exists(os.path.join("logs", f"training_log_{date}.csv")):
                with open(os.path.join("logs", f"training_log_{date}.csv"), "w") as f:
                    f.write("Epoch,Train Loss,Train Acc,Val Acc,Train MCC,Val MCC,Val AUC,Val AUPRC,TP,TN,FP,FN,Epoch Time (s)\n")
            with open(os.path.join("logs", f"training_log_{date}.csv"), "a") as f:
                f.write(f"{epoch+1},{train_loss:.6f},{train_acc:.4f},{val_acc:.4f},{train_mcc:.4f},{val_mcc:.4f},{val_auc:.4f},{val_auprc:.4f},{TP},{TN},{FP},{FN},{epoch_time:.2f}\n")

            # if no_improve_epochs >= 15:
            #     print("No improvement in validation accuracy for 10 epochs, stopping early.")
            #     break
        
        self.evaluate(self.test_dl)

        # TODO: whole bunch incomplete (is this still true)


    def evaluate(self, data_loader):
        self.model.eval()
        uniprot_ids, drug_ids, pred_pairs, conf_list, conf_list, prob_list, ev_list, bk_list = self.predict_test(data_loader)

        val_results = np.squeeze(np.array(pred_pairs))  # [N, 2]
        val_acc = 100 * np.equal(val_results[:, 0], val_results[:, 1]).sum() / len(val_results)
        val_mcc = matthews_corrcoef(val_results[:, 1], val_results[:, 0])
        val_auc = roc_auc_score(val_results[:, 1], prob_list)
        val_auprc = average_precision_score(val_results[:, 1], prob_list)
        print('Test accuracy: %.4f%% MCC: %.4f AUC: %.4f AUPRC: %.4f' % (val_acc, val_mcc, val_auc, val_auprc))
        val_f1 = f1_score(val_results[:, 1], val_results[:, 0])
        val_precision = precision_score(val_results[:, 1], val_results[:, 0])
        val_recall = recall_score(val_results[:, 1], val_results[:, 0])
        print('F1: %.4f Precision: %.4f Recall: %.4f' % (val_f1, val_precision, val_recall))

        # plot ROC curve
        fpr, tpr, thresholds = roc_curve(val_results[:, 1], prob_list, pos_label=1)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %.4f)' % val_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()

    def save_model(self, epoch, dir="saved"):
        os.makedirs(dir, exist_ok=True)
        ckpt = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optim.state_dict(),
            'cfg': self.cfg
        }
        torch.save(ckpt, os.path.join(dir, f"model_{self.id}_epoch_{epoch}.pt"))
    # Additional methods for training, validation, testing would go here

if __name__ == "__main__":
    cfg = get_cfg_defaults()
    model = Model(cfg)
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    solver = Solver(model, cfg, device=device,
                    optim=torch.optim.Adam, loss_fn=cfg.SOLVER.LOSS_FN, eval=None)

    solver.train(solver.train_dl, solver.val_dl)
    # evaluate on best model (starting epoch - 10, or latest entry in best_models)

    # dirichlet loss test   
    # B = 4
    # K = 2
    # alpha = torch.tensor([[1600000, 1.5], [1.6, 1.5], [1.8, 1.7] ,[1.4, 1.3]])
    # print("Alpha:", alpha)
    # labels = torch.tensor([0, 0, 1, 0])
    # loss = dirichlet_loss(torch.tensor([[1.5, 1.5], [1.5, 1.5], [1.5, 1.5] ,[1.5, 1.5]]), labels)
    # print("Dirichlet loss test:", loss.item())

    # print("labels shape:", labels.shape)
    # print("labels dtype:", labels.dtype)
    # print("labels unique:", labels.unique())
