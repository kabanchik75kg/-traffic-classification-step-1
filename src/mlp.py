# src/mlp.py
# MLP-классификатор (PyTorch, CPU) для flow-baseline.
# Обучается минибатчами на linear-view (80 признаков после препроцессора),
# с честным early stopping по val-метрике и откатом к лучшим весам.

import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score  # type: ignore

from src import config, data


class MLP(nn.Module):
    """Простой 3-слойный персептрон: 80 → 128 → 64 → 1 (logit)."""
    def __init__(self, n_in, hidden=(128, 64), p_drop=0.2):
        super().__init__()
        layers, prev = [], n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(p_drop)]
            prev = h
        layers += [nn.Linear(prev, 1)]   # один logit, BCEWithLogitsLoss
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


def _val_pr_auc(model, pre, feature_cols, device):
    """PR-AUC на val (для early stopping). Прогон по батчам, без градиентов."""
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for X_b, y_b in data.iter_batches(config.VAL_PATH, feature_cols):
            Xt = torch.from_numpy(pre.transform(X_b)).to(device)
            p = torch.sigmoid(model(Xt)).cpu().numpy()
            ps.append(p)
            ys.append(y_b)
    return average_precision_score(np.concatenate(ys), np.concatenate(ps))


def train_mlp(pre, feature_cols, *,
              epochs=30, lr=1e-3, weight_decay=1e-5,
              patience=4, pos_weight=None, seed=config.RANDOM_STATE,
              device="cpu", verbose=True):
    """
    Обучает MLP с early stopping по val PR-AUC.
    pos_weight — вес положительного класса для BCEWithLogitsLoss
    (дисбаланс 73/27).
    Возвращает обученную модель (с лучшими весами).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_in = pre.n_features_out_
    model = MLP(n_in).to(device)
    opt = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    pw = torch.tensor([pos_weight], dtype=torch.float32, device=device) \
        if pos_weight is not None else None
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)

    best_score, best_state, no_improve = -1.0, None, 0

    for epoch in range(1, epochs + 1):
        model.train()
        t0, n_batches = time.time(), 0
        for X_b, y_b in data.iter_batches(config.TRAIN_PATH, feature_cols):
            Xt = torch.from_numpy(pre.transform(X_b)).to(device)
            yt = torch.from_numpy(y_b.astype(np.float32)).to(device)
            opt.zero_grad()
            loss = loss_fn(model(Xt), yt)
            loss.backward()
            opt.step()
            n_batches += 1

        val_score = _val_pr_auc(model, pre, feature_cols, device)
        improved = val_score > best_score + 1e-5
        if improved:
            best_score = val_score
            best_state = {
                k: v.cpu().clone()
                for k, v in model.state_dict().items()
            }
            no_improve = 0
        else:
            no_improve += 1

        if verbose:
            mark = "  ✓ best" if improved else ""
            print(f"  эпоха {epoch:>2}/{epochs}  val PR-AUC={val_score:.5f}  "
                  f"({time.time()-t0:.0f}s){mark}")

        if no_improve >= patience:
            if verbose:
                print(f"  early stopping: {patience} эпох без улучшения.")
            break

    model.load_state_dict(best_state)   # откат к лучшим весам
    if verbose:
        print(f"Лучший val PR-AUC: {best_score:.5f}")
    return model


def make_predict_fn(model, pre, device="cpu"):
    """Оборачивает MLP в predict_proba_fn(X_raw)->proba для evaluate."""
    def predict(X_raw):
        model.eval()
        with torch.no_grad():
            Xt = torch.from_numpy(pre.transform(X_raw)).to(device)
            return torch.sigmoid(model(Xt)).cpu().numpy()
    return predict
