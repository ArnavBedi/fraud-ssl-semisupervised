import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

class PUElkanNoto:
    """
    Positive–Unlabeled learning (Elkan–Noto):
      1) Train a classifier on S=1 (labeled positives) vs S=0 (unlabeled).
      2) Estimate c = P(S=1 | Y=1) using predicted probs on labeled positives.
      3) PU posterior: p(Y=1|x) = p(S=1|x) / c, clipped to [0,1].
    """
    def __init__(self, max_iter=200):
        self.base = LogisticRegression(max_iter=max_iter, class_weight="balanced")
        self.clf = CalibratedClassifierCV(self.base, cv=3, method='sigmoid')
        self.c_ = 0.5

    def fit(self, X, s):
        # s: 1 for labeled positive, 0 for unlabeled
        self.clf.fit(X, s)
        pos_idx = (s == 1)
        if pos_idx.any():
            ps = self.clf.predict_proba(X[pos_idx])[:, 1]
            self.c_ = float(np.clip(ps.mean(), 1e-3, 1.0))
        else:
            self.c_ = 0.5
        return self

    def predict_proba(self, X):
        ps = self.clf.predict_proba(X)[:, 1]
        pu = np.clip(ps / self.c_, 0.0, 1.0)
        return np.vstack([1 - pu, pu]).T
