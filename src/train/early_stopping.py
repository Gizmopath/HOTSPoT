class EarlyStopping:
    def __init__(self, patience=5, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_val_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.counter = 0
            if self.verbose:
                print(f"Validation loss improved to {val_loss:.4f}.")
        else:
            self.counter += 1
            if self.verbose:
                print(f"No improvement for {self.counter} epoch(s).")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("Early stopping triggered.")
