from CONSTANTS import *
from util import logger

logger = logger('EarlyStopping')


class EarlyStoppingF1:
    def __init__(self, patience=5, delta=0.0, accuracy_threshold=0.999, verbose=False):
        self.patience = patience
        self.delta = delta
        self.accuracy_threshold = accuracy_threshold
        self.verbose = verbose
        self.best_f1 = -np.inf
        self.counter = 0
        self.early_stop = False

    def __call__(self, current_f1):
        if current_f1 < 0.9:
            return
        if current_f1 > self.accuracy_threshold:
            self.early_stop = True
        elif current_f1 > self.best_f1 + self.delta:
            if self.verbose:
                logger.info(f"F1 score improves: {self.best_f1:.4f} → {current_f1:.4f}")
            self.best_f1 = current_f1
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                logger.info(f"EarlyStoppingF1 counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
