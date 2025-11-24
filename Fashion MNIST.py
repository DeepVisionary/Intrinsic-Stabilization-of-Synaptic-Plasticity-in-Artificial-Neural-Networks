# -*- coding: utf-8 -*-

# fashion_mnist_td_vs_baseline.py
import copy, random, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# =========================
# Config / Hyperparameters
# =========================
SEED           = 42
EPOCHS         = 30
BATCH_SIZE     = 256
LR             = 3e-3          # main & baseline LR
WEIGHT_DECAY   = 0.0           # Adam usually fine with 0 for this scale
TAU            = 1e-2          # predictive head LR (constant)
SIGMA_CONST    = 1e-2         # constant TD mix (set 0.0 to disable TD)
T_KL           = 1.0
NUM_WORKERS    = 0             # keep 0 for Windows/Spyder
DATA_ROOT      = "./data"
LOG_EVERY      = 200
PLOT           = True

# ================
# Repro & Device
# ================
def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
set_seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# =========
# Data (Fashion-MNIST)
# =========
# Official-ish channel stats (computed over the train set)
FASHION_MEAN = (0.2860,)
FASHION_STD  = (0.3530,)

tf_train = transforms.Compose([
    # Light aug helps a bit; comment out if you want pure baseline
    transforms.RandomCrop(28, padding=2),
    transforms.RandomHorizontalFlip(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(FASHION_MEAN, FASHION_STD),
])

tf_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(FASHION_MEAN, FASHION_STD),
])

full_train = datasets.FashionMNIST(root=DATA_ROOT, train=True,  download=True, transform=tf_train)
test_ds    = datasets.FashionMNIST(root=DATA_ROOT, train=False, download=True, transform=tf_test)

# Train/val split for early stopping-like monitoring (50k/10k)
train_size = 50_000
val_size   = len(full_train) - train_size
split_gen  = torch.Generator().manual_seed(SEED)
train_ds, val_ds = random_split(full_train, [train_size, val_size], generator=split_gen)

def make_loader(dataset, batch_size=BATCH_SIZE, shuffle=True):
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=NUM_WORKERS,
                      pin_memory=(device == "cuda"),
                      persistent_workers=False)

train_loader = make_loader(train_ds, shuffle=True)
val_loader   = make_loader(val_ds,   shuffle=False)
test_loader  = make_loader(test_ds,  shuffle=False)

# =======================
# Model & TD Components
# =======================
class SmallCNN(nn.Module):
    """
    Tiny CNN for 1x28x28 Fashion-MNIST:
    Conv(32)->ReLU->MaxPool
    Conv(64)->ReLU->MaxPool
    Flatten->FC(128)->ReLU->FC(10)
    Exposes `.last` so we can inject TD grads only at the output layer.
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                 # 32 x 14 x 14
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                 # 64 x 7 x 7
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128), nn.ReLU()
        )
        self.last = nn.Linear(128, num_classes)      # logits layer

    def forward(self, x):
        h = self.features(x)
        h = self.head(h)
        z = self.last(h)                             # logits
        yhat = F.softmax(z, dim=-1)
        return z, yhat

class PredictiveHead(nn.Module):
    """Tiny head mapping logits -> logits (C -> C)."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.lin = nn.Linear(num_classes, num_classes)
    def forward(self, z):
        return self.lin(z)

def kl_forward(p_logit, q_logit, T=1.0):
    """
    Forward KL: KL( softmax(p/T) || softmax(q/T) )
    """
    p = F.log_softmax(p_logit / T, dim=-1).exp()
    logp = torch.log(p + 1e-12)
    logq = F.log_softmax(q_logit / T, dim=-1)
    return torch.sum(p * (logp - logq), dim=-1).mean()

# ===============
# Eval helpers
# ===============
@torch.no_grad()
def eval_loss_acc(model, loader):
    model.eval()
    total, correct = 0, 0
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        z, _ = model(x)
        loss = F.cross_entropy(z, y, reduction='sum')
        total_loss += loss.item()
        pred = z.argmax(dim=1)
        correct += (pred == y).sum().item()
        total   += y.numel()
    model.train()
    return total_loss / max(1, total), correct / max(1, total)

# ===================================
# Side-by-side training on same batch
# ===================================
def train_side_by_side(model_base,
                       model_td,
                       epochs=EPOCHS,
                       lr=LR,
                       tau=TAU,
                       sigma_const=SIGMA_CONST,
                       T_kl=T_KL,
                       weight_decay=WEIGHT_DECAY,
                       log_every=LOG_EVERY,
                       log_td_metrics=True):
    """
    Train baseline (CE only) and TD (CE + σ·TD on output layer) models step-by-step
    on the SAME batches. Returns test loss/acc curves for plotting.
    """
    model_base.to(device); model_td.to(device)
    pred_head = PredictiveHead(num_classes=10).to(device)

    opt_base = torch.optim.Adam(model_base.parameters(), lr=lr, weight_decay=weight_decay)
    opt_main = torch.optim.Adam(model_td.parameters(),   lr=lr, weight_decay=weight_decay)
    opt_pred = torch.optim.Adam(pred_head.parameters(),  lr=tau)  # usually no wd on head

    history = {
        "test_loss_base": [], "test_acc_base": [],
        "test_loss_td":   [], "test_acc_td":   [],
        "val_loss_base":  [], "val_acc_base":  [],
        "val_loss_td":    [], "val_acc_td":    [],
    }

    global_step = 0
    for ep in range(1, epochs + 1):
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            # ---- Baseline update (CE only) ----
            opt_base.zero_grad(set_to_none=True)
            zb, _ = model_base(x)
            Lb = F.cross_entropy(zb, y)
            Lb.backward()
            opt_base.step()

            # ---- TD model update (CE + output-only TD) ----
            sigma = sigma_const

            opt_main.zero_grad(set_to_none=True)
            opt_pred.zero_grad(set_to_none=True)

            zt, _ = model_td(x)
            L_sup = F.cross_entropy(zt, y)

            # predictive head learns from current logits (teacher = zt)
            pred_logits = pred_head(zt.detach())
            L_td_head   = kl_forward(pred_logits, zt.detach(), T=T_kl)

            # TD term to be applied only to the last layer of model_td
            L_td_out = kl_forward(zt, pred_logits.detach(), T=T_kl)

            # supervised grads through whole model_td
            L_sup.backward(retain_graph=True)

            # snapshot supervised grads of last layer BEFORE TD injection
            W, b   = model_td.last.weight, model_td.last.bias
            sup_gW = W.grad.detach().clone() if W.grad is not None else torch.zeros_like(W)
            sup_gb = b.grad.detach().clone() if b.grad is not None else torch.zeros_like(b)

            # update predictive head (τ) – no effect on model_td
            L_td_head.backward()
            opt_pred.step()

            # TD grads ONLY for last-layer params
            td_gW, td_gb = torch.autograd.grad(L_td_out, [W, b], retain_graph=False)

            # optional diagnostics
            if log_td_metrics and (global_step % log_every == 0):
                tdW, supW = td_gW.norm().item(), sup_gW.norm().item() + 1e-12
                tdb, supb = td_gb.norm().item(), sup_gb.norm().item() + 1e-12
                cosW = F.cosine_similarity(td_gW.flatten(), sup_gW.flatten(), dim=0).item()
                cosb = F.cosine_similarity(td_gb.flatten(), sup_gb.flatten(), dim=0).item()
                print(f"[step {global_step:6d}] σ={sigma:.3e}  "
                      f"TD/W: norm={tdW:.3e} ratio={tdW/supW:.3f} cos={cosW:+.3f}  "
                      f"TD/b: norm={tdb:.3e} ratio={tdb/supb:.3f} cos={cosb:+.3f}")

            # ▶ inject σ·TD into last-layer grads (add to supervised grads)
            with torch.no_grad():
                if W.grad is None: W.grad = torch.zeros_like(W)
                if b.grad is None: b.grad = torch.zeros_like(b)
                W.grad.add_(sigma * td_gW)
                b.grad.add_(sigma * td_gb)

            opt_main.step()
            global_step += 1

        # end-epoch: evaluate on VAL and TEST for plotting
        val_loss_b, val_acc_b = eval_loss_acc(model_base, val_loader)
        val_loss_t, val_acc_t = eval_loss_acc(model_td,   val_loader)
        test_loss_b, test_acc_b = eval_loss_acc(model_base, test_loader)
        test_loss_t, test_acc_t = eval_loss_acc(model_td,   test_loader)

        history["val_loss_base"].append(val_loss_b); history["val_acc_base"].append(val_acc_b)
        history["val_loss_td"].append(val_loss_t);   history["val_acc_td"].append(val_acc_t)
        history["test_loss_base"].append(test_loss_b); history["test_acc_base"].append(test_acc_b)
        history["test_loss_td"].append(test_loss_t);   history["test_acc_td"].append(test_acc_t)

        print(f"Epoch {ep}/{epochs} | "
              f"val(base):  loss={val_loss_b:.4f} acc={val_acc_b*100:.2f}%   "
              f"val(TD):    loss={val_loss_t:.4f} acc={val_acc_t*100:.2f}% | "
              f"test(base): loss={test_loss_b:.4f} acc={test_acc_b*100:.2f}%   "
              f"test(TD):   loss={test_loss_t:.4f} acc={test_acc_t*100:.2f}%")

    return history

# =========
# Main
# =========
def main():
    # identical init for fairness
    base = SmallCNN(num_classes=10).to(device)
    td   = SmallCNN(num_classes=10).to(device)
    td.load_state_dict(copy.deepcopy(base.state_dict()))

    hist = train_side_by_side(
        model_base=base,
        model_td=td,
        epochs=EPOCHS,
        lr=LR,
        tau=TAU,
        sigma_const=SIGMA_CONST,   # set to 0.0 to disable TD
        T_kl=T_KL,
        weight_decay=WEIGHT_DECAY,
        log_every=LOG_EVERY,
        log_td_metrics=True
    )

    if PLOT:
        epochs = np.arange(1, EPOCHS + 1)

        plt.figure()
        plt.plot(epochs, 100*np.array(hist["test_acc_base"]), label="Baseline")
        plt.plot(epochs, 100*np.array(hist["test_acc_td"]),   label="TD (output-only)")
        plt.xlabel("Epoch"); plt.ylabel("Test Accuracy (%)"); plt.title("Fashion-MNIST Test Accuracy")
        plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

        plt.figure()
        plt.plot(epochs, hist["test_loss_base"], label="Baseline")
        plt.plot(epochs, hist["test_loss_td"],   label="TD (output-only)")
        plt.xlabel("Epoch"); plt.ylabel("Test Loss (CE)"); plt.title("Fashion-MNIST Test Loss")
        plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()

