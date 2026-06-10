import torch
@torch.no_grad()
def accuracy(logits, targets):
    preds = torch.argmax(logits, dim=1)
    return (preds == targets).float().mean().item()

class AverageMeter:
    def __init__(self):
        self.sum = 0.0; self.cnt = 0
    def update(self, val, n=1):
        self.sum += float(val) * n; self.cnt += n
    @property
    def avg(self): return self.sum / max(self.cnt,1)
