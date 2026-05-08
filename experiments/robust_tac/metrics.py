import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def clustering_accuracy(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    true_values = {label: idx for idx, label in enumerate(sorted(set(y_true.tolist())))}
    pred_values = {label: idx for idx, label in enumerate(sorted(set(y_pred.tolist())))}
    true_idx = np.array([true_values[x] for x in y_true], dtype=np.int64)
    pred_idx = np.array([pred_values[x] for x in y_pred], dtype=np.int64)
    dim = max(len(true_values), len(pred_values))
    matrix = np.zeros((dim, dim), dtype=np.int64)
    for pred, true in zip(pred_idx, true_idx):
        matrix[pred, true] += 1
    row_ind, col_ind = linear_sum_assignment(matrix.max() - matrix)
    return float(matrix[row_ind, col_ind].sum() / max(1, y_true.size))


def compute_cluster_metrics(labels, clean_preds, adv_preds):
    labels = np.asarray(labels)
    clean_preds = np.asarray(clean_preds)
    adv_preds = np.asarray(adv_preds)
    return {
        "clean_acc": clustering_accuracy(labels, clean_preds),
        "clean_nmi": float(normalized_mutual_info_score(labels, clean_preds)),
        "clean_ari": float(adjusted_rand_score(labels, clean_preds)),
        "adv_acc": clustering_accuracy(labels, adv_preds),
        "adv_nmi": float(normalized_mutual_info_score(labels, adv_preds)),
        "adv_ari": float(adjusted_rand_score(labels, adv_preds)),
        "cfr": float(np.mean(clean_preds != adv_preds)),
    }
