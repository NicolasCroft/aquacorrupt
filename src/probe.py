"""
Linear probe = the only thing that trains. Fit a standardizer + logistic regression on
clean TRAIN features, then measure accuracy on clean TEST and on corrupted TEST at each
severity. Robustness of the *representation* shows up as how slowly accuracy falls.

We report, per backbone:
  clean_acc               accuracy on clean test
  acc[severity]           accuracy at each corruption severity
  retention[severity]     acc[severity] / clean_acc   (1.0 = perfectly robust)
  mean_retention          average retention across severities (single headline number)
"""
from __future__ import annotations
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from . import cache as F


def fit_probe(Xtr, ytr, seed=0):
    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
    clf.fit(scaler.transform(Xtr), ytr)
    return scaler, clf


def accuracy(scaler, clf, X, y):
    return float((clf.predict(scaler.transform(X)) == y).mean())


def evaluate_backbone(emb_dir, backbone, corruption, severities, seed=0):
    Xtr, ytr = F.load(emb_dir, backbone, "train", "clean", 0)
    Xte, yte = F.load(emb_dir, backbone, "test", "clean", 0)
    scaler, clf = fit_probe(Xtr, ytr, seed=seed)

    clean_acc = accuracy(scaler, clf, Xte, yte)
    acc, retention = {}, {}
    for s in severities:
        Xs, ys = F.load(emb_dir, backbone, "test", corruption, s)
        a = accuracy(scaler, clf, Xs, ys)
        acc[s] = a
        retention[s] = a / clean_acc if clean_acc > 0 else 0.0

    return {
        "backbone": backbone,
        "clean_acc": clean_acc,
        "acc": acc,
        "retention": retention,
        "mean_retention": float(np.mean(list(retention.values()))) if retention else 0.0,
    }
