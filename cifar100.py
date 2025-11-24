# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 14:53:07 2025

@author: tommy
"""

# cifar100_resnet18_td_vs_baseline.py
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
EPOCHS         = 200
BATCH_SIZE     = 256
LR             = 1e-3        # main & baseline LR
WEIGHT_DECAY   = 5e-4
TAU            = 1e-3        # predictive head LR (constant)
SIGMA_CONST    = 1e-1        # constant TD mix (set 0.0 to disable TD)
T_KL           = 1.0
NUM_WORKERS    = 0           # keep 0 for Windows/Spyder
DATA_ROOT      = "./data"
LOG_EVERY      = 200

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
# Data (CIFAR-100)
# =========
C100_MEAN = (0.5071, 0.4867, 0.4408)
C100_STD  = (0.2675, 0.2565, 0.2761)

tf_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    # If available in your torchvision, you can also try:
    # transforms.RandAugment(num_ops=2, magnitude=9),
    transforms.ToTensor(),
    transforms.Normalize(C100_MEAN, C100_STD),
])

tf_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(C100_MEAN, C100_STD),
])

full_train = datasets.CIFAR100(root=DATA_ROOT, train=True,  download=True, transform=tf_train)
test_ds    = datasets.CIFAR100(root=DATA_ROOT, train=False, download=True, transform=tf_test)

train_size = 45_000
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
# Model: ResNet-18 (CIFAR stem)
# =======================
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)

        self.downsample = None
        if stride != 1 or in_planes != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion)
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(out + identity)
        return out

def _make_layer(block, in_planes, planes, blocks, stride):
    layers = [block(in_planes, planes, stride)]
    for _ in range(1, blocks):
        layers.append(block(planes * block.expansion, planes, 1))
    return nn.Sequential(*layers)

class ResNet18CIFAR(nn.Module):
    """
    ResNet-18 adapted for CIFAR (32x32):
    - 3x3 conv stem (stride=1)
    - Residual layers with strides [1,2,2,2]
    - Global pool + linear classifier
    Exposes `.last` as output layer so TD wiring stays unchanged.
    """
    def __init__(self, num_classes=100):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)

        self.layer1 = _make_layer(BasicBlock, 64,  64,  2, stride=1)  # 32x32
        self.layer2 = _make_layer(BasicBlock, 64,  128, 2, stride=2)  # 16x16
        self.layer3 = _make_layer(BasicBlock, 128, 256, 2, stride=2)  # 8x8
        self.layer4 = _make_layer(BasicBlock, 256, 512, 2, stride=2)  # 4x4

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.last    = nn.Linear(512, num_classes)  # final logits for TD

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x).flatten(1)
        z = self.last(x)
        return z, F.softmax(z, dim=-1)

# =======================
# TD Head & KL
# =======================
class PredictiveHead(nn.Module):
    """Tiny head mapping logits -> logits (C -> C)."""
    def __init__(self, num_classes=100):
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
    pred_head = PredictiveHead(num_classes=100).to(device)

    opt_base = torch.optim.Adam(model_base.parameters(), lr=lr, weight_decay=weight_decay)
    opt_main = torch.optim.Adam(model_td.parameters(),   lr=lr, weight_decay=weight_decay)
    opt_pred = torch.optim.Adam(pred_head.parameters(),  lr=tau)  # usually no wd on head

    history = {
        "test_loss_base": [], "test_acc_base": [],
        "test_loss_td":   [], "test_acc_td":   []
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
            sigma = sigma_const  # constant σ

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

        # end-epoch: evaluate on TEST set for plotting
        test_loss_b, test_acc_b = eval_loss_acc(model_base, test_loader)
        test_loss_t, test_acc_t = eval_loss_acc(model_td,   test_loader)
        history["test_loss_base"].append(test_loss_b)
        history["test_acc_base"].append(test_acc_b)
        history["test_loss_td"].append(test_loss_t)
        history["test_acc_td"].append(test_acc_t)

        print(f"Epoch {ep}/{epochs} | "
              f"test(base): loss={test_loss_b:.4f} acc={test_acc_b*100:.2f}%   "
              f"test(TD): loss={test_loss_t:.4f} acc={test_acc_t*100:.2f}%")

    return history

# =========
# Main
# =========
def main():
    # identical init
    base = ResNet18CIFAR(num_classes=100).to(device)
    td   = ResNet18CIFAR(num_classes=100).to(device)
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

    # ---- Plots: Test Accuracy & Test Loss ----
    epochs = np.arange(1, EPOCHS + 1)

    plt.figure()
    plt.plot(epochs, 100*np.array(hist["test_acc_base"]), label="Baseline")
    plt.plot(epochs, 100*np.array(hist["test_acc_td"]),   label="TD (output-only)")
    plt.xlabel("Epoch"); plt.ylabel("Test Accuracy (%)"); plt.title("CIFAR-100 Test Accuracy (ResNet-18)")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(epochs, hist["test_loss_base"], label="Baseline")
    plt.plot(epochs, hist["test_loss_td"],   label="TD (output-only)")
    plt.xlabel("Epoch"); plt.ylabel("Test Loss (CE)"); plt.title("CIFAR-100 Test Loss (ResNet-18)")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
