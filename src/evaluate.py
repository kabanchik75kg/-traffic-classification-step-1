# src/evaluate.py
# Единая оценка моделей.
# Используется всеми обучающими ноутбуками и ноутбуком test.
#
# Дисциплина порога (критично для честности):
#   threshold=None  → режим VAL: порог подбирается по PR-кривой и ВОЗВРАЩАЕТСЯ.
#   threshold=float → режим TEST: используется готовый порог,
#      замороженный на val.
# Это исключает утечку выбора порога в тестовую метрику.
#
# Threshold-независимые метрики (PR-AUC, ROC-AUC, FPR@TPR95)
# от порога не зависят и считаются всегда честно.

import gc
import json
import time

import numpy as np
from sklearn.metrics import (  # type: ignore
    roc_curve, precision_recall_curve, confusion_matrix,
    f1_score, roc_auc_score, average_precision_score, matthews_corrcoef,
)

from src import config, data


def _collect_predictions(predict_proba_fn, path, feature_cols):
    """Прогон по батчам → (y_true, y_proba, lbl) для всего файла."""
    y_chunks, p_chunks, lbl_chunks = [], [], []
    t0 = time.time()
    batches = data.iter_batches(path, feature_cols, with_label=True)
    for X_b, y_b, lbl_b in batches:
        p_chunks.append(predict_proba_fn(X_b).astype(np.float32))
        y_chunks.append(y_b)
        lbl_chunks.append(lbl_b)
        del X_b
        gc.collect()
    y_true = np.concatenate(y_chunks)
    y_proba = np.concatenate(p_chunks)
    lbl = np.concatenate(lbl_chunks)
    del y_chunks, p_chunks, lbl_chunks
    gc.collect()
    return y_true, y_proba, lbl, time.time() - t0


def evaluate(model_name, predict_proba_fn, path, feature_cols, *,
             threshold=None, split_name="test", save=True, verbose=True):
    """
    Оценивает модель на одном сплите.

    threshold=None → подбирает оптимум по PR-кривой (режим val),
    возвращает его в метриках.
    threshold=float → применяет замороженный порог (режим test).

    Возвращает dict метрик.
    При save=True пишет results/baseline/{model_name}_{split}.json.
    """
    y_true, y_proba, lbl, infer_t = _collect_predictions(
        predict_proba_fn, path, feature_cols
    )

    # ── Threshold-независимые метрики ──
    pr_auc = float(average_precision_score(y_true, y_proba))
    roc_auc = float(roc_auc_score(y_true, y_proba))
    fpr_arr, tpr_arr, _ = roc_curve(y_true, y_proba)
    fpr_at_95 = float(fpr_arr[tpr_arr >= 0.95].min())

    # ── Порог: подбор (val) или заморозка (test) ──
    if threshold is None:
        prec, rec, thr = precision_recall_curve(y_true, y_proba)
        f1_arr = 2 * prec * rec / (prec + rec + 1e-10)
        best_idx = int(np.argmax(f1_arr))
        used_threshold = float(thr[best_idx]) if best_idx < len(thr) else 0.5
        threshold_mode = "tuned_on_this_split"
    else:
        used_threshold = float(threshold)
        threshold_mode = "frozen_from_val"

    y_pred = (y_proba >= used_threshold).astype(np.int8)

    # ── Порог-зависимые метрики ──
    f1_macro = float(f1_score(y_true, y_pred, average="macro"))
    mcc = float(matthews_corrcoef(y_true, y_pred))

    tn, fp, fn, tp = (int(x) for x in confusion_matrix(y_true, y_pred).ravel())
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    precision_v = tp / (tp + fp) if (tp + fp) else 0.0
    recall_v = tp / (tp + fn) if (tp + fn) else 0.0

    # ── Per-class recall ──
    per_class = {}
    for cls in np.unique(lbl):
        mask = lbl == cls
        n = int(mask.sum())
        r = float((y_pred[mask] == 0).mean()) if cls == config.BENIGN_LABEL \
            else float((y_pred[mask] == 1).mean())
        per_class[str(cls)] = {"n": n, "recall": round(r, 4)}

    metrics = {
        "model":            model_name,
        "split":            split_name,
        "n_rows":           int(len(y_true)),
        "inference_sec":    round(infer_t, 2),
        "threshold":        round(used_threshold, 4),
        "threshold_mode":   threshold_mode,
        "F1_macro":         round(f1_macro, 4),
        "MCC":              round(mcc, 4),
        "PR_AUC":           round(pr_auc, 6),
        "ROC_AUC":          round(roc_auc, 6),
        "FPR_at_TPR95":     round(fpr_at_95, 6),
        "binary": {
            "precision": round(precision_v, 4),
            "recall":    round(recall_v, 4),
            "FPR":       round(fpr, 6),
            "FNR":       round(fnr, 6),
        },
        "confusion": {"TN": tn, "FP": fp, "FN": fn, "TP": tp},
        "per_class_recall": per_class,
    }

    if verbose:
        _print_report(metrics)
    if save:
        out = config.BASELINE_DIR / f"{model_name}_{split_name}.json"
        with open(out, "w") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        if verbose:
            print(f"\n→ сохранено: {out}")

    return metrics


def _print_report(m):
    print(f"\n{'='*60}\n  {m['model']}  ({m['split']})\n{'='*60}")
    print(f"строк {m['n_rows']:,} за {m['inference_sec']}s  |  "
          f"порог {m['threshold']} ({m['threshold_mode']})")
    print(f"F1_macro {m['F1_macro']}  MCC {m['MCC']}  "
          f"PR-AUC {m['PR_AUC']}  ROC-AUC {m['ROC_AUC']}  "
          f"FPR@95 {m['FPR_at_TPR95']}"
          )
    c = m["confusion"]
    print(f"TN={c['TN']:,}  FP={c['FP']:,}  FN={c['FN']:,}  TP={c['TP']:,}  "
          f"(FPR={m['binary']['FPR']:.4f}, FNR={m['binary']['FNR']:.4f})")
    print("\nper-class recall:")
    per_class = sorted(m["per_class_recall"].items(), key=lambda x: -x[1]["n"])
    for cls, d in per_class:
        flag = " ◄" if d["recall"] < 0.5 and cls != config.BENIGN_LABEL else ""
        print(f"  {cls:<32} n={d['n']:>8,}  recall={d['recall']:.2%}{flag}")
