import os
import torch
import pandas as pd
import warnings
from sklearn.metrics import matthews_corrcoef, confusion_matrix, f1_score, recall_score, precision_score, accuracy_score, roc_auc_score, precision_recall_curve, auc, roc_curve
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import matplotlib.pyplot as plt

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
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, mode='min', factor=0.5, patience=5)  # FIXME

    def predict(self, data_loader, epoch, optim):
        running_loss = 0.0
        for i, batch in enumerate(data_loader):
            metadata = batch.metadata
            label_org = batch.l.to(self.device)  # (B)

            # pad to max length in batch
            sequence_lengths = metadata['length'][:, None].to(self.device)  # [batchsize, 1]
            max_len = sequence_lengths.max().item()

            loss = self.loss_fn()

            if optim:  # run backpropagation if an optimizer is provided
                loss.backward()
                self.optim.step()
                self.optim.zero_grad()

            running_loss += loss.item()

            if i % 100 == 99:  # log every log_iterations
                if epoch:
                    print('Epoch %d ' % (epoch), end=' ')
                    print('[Iter %5d/%5d] %s: loss: %.7f, accuracy: %.4f%%' % (
                        i + 1, len(data_loader), 'Train' if optim else 'Val', loss.item(),
                        100 * (pred==label_org).sum().item() / self.batch_size))
        return None, None

    def train(self, train_loader, val_loader):
        no_improve_epochs = 0
        for epoch in range(self.epochs):
            self.model.train()
            train_loss, train_results = self.predict(val_loader, epoch + 1)

    def evaluate(self, data_loader):
        pass

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
    # Additional methods for training, validation, testing would go here