from river.drift import ADWIN

class DriftWatcher:
    def __init__(self, delta=0.002):
        self.adwin = ADWIN(delta=delta)
        self.drift_points = []

    def update(self, score, t):
        """
        score: float anomaly/risk score
        t:     integer/time index
        returns True if drift detected at this update
        """
        changed = self.adwin.update(score)
        if changed:
            self.drift_points.append(t)
        return changed
