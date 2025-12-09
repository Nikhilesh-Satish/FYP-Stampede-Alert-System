# line_counter.py
import numpy as np
from scipy.ndimage import maximum_filter, gaussian_filter
from collections import OrderedDict

class CentroidTracker:
    """
    Simple centroid tracker using nearest-neighbor matching.
    Keeps object centroids and short history to detect crossings.
    """
    def __init__(self, max_lost=10, max_distance=60):
        self.next_object_id = 0
        self.objects = OrderedDict()   # id -> (x,y)
        self.lost = OrderedDict()      # id -> frames lost
        self.history = {}              # id -> list of (x,y) positions
        self.max_lost = max_lost
        self.max_distance = max_distance

    def register(self, centroid):
        oid = self.next_object_id
        self.next_object_id += 1
        self.objects[oid] = centroid
        self.lost[oid] = 0
        self.history[oid] = [centroid]
        return oid

    def deregister(self, oid):
        if oid in self.objects: del self.objects[oid]
        if oid in self.lost: del self.lost[oid]
        if oid in self.history: del self.history[oid]

    def update(self, detections):
        """
        detections: list of (x,y) tuples
        returns: dict id -> centroid (current)
        """
        if len(detections) == 0:
            # increment lost counters
            for oid in list(self.lost.keys()):
                self.lost[oid] += 1
                if self.lost[oid] > self.max_lost:
                    self.deregister(oid)
            return self.objects

        dets = np.array(detections)
        if len(self.objects) == 0:
            for i in range(len(dets)):
                self.register(tuple(dets[i]))
            return self.objects

        object_ids = list(self.objects.keys())
        object_centroids = np.array(list(self.objects.values()))

        # distance matrix (objects x detections)
        D = np.linalg.norm(object_centroids[:, None, :] - dets[None, :, :], axis=2)

        # greedy matching
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        assigned_rows, assigned_cols = set(), set()
        for r, c in zip(rows, cols):
            if r in assigned_rows or c in assigned_cols:
                continue
            if D[r, c] > self.max_distance:
                continue
            oid = object_ids[r]
            new_centroid = tuple(dets[c])
            self.objects[oid] = new_centroid
            self.lost[oid] = 0
            self.history.setdefault(oid, []).append(new_centroid)
            assigned_rows.add(r)
            assigned_cols.add(c)

        # register unassigned detections
        for i in range(len(dets)):
            if i not in assigned_cols:
                self.register(tuple(dets[i]))

        # mark unassigned existing objects as lost
        for i, oid in enumerate(object_ids):
            if i not in assigned_rows:
                self.lost[oid] += 1
                if self.lost[oid] > self.max_lost:
                    self.deregister(oid)

        return self.objects

def extract_peaks_from_density(den_map, min_distance=8, threshold_rel=0.12, sigma=1.0):
    """
    den_map: 2D numpy array (H, W) - assumed upsampled to frame size
    returns: list of (x, y) centroids (integers)
    """
    if den_map is None:
        return []

    # smooth to reduce spurious peaks
    den_s = gaussian_filter(den_map, sigma=sigma)

    neighborhood_size = max(3, int(min_distance))
    local_max = (den_s == maximum_filter(den_s, size=neighborhood_size))
    threshold = den_s.max() * threshold_rel
    peaks_mask = local_max & (den_s >= threshold)

    ys, xs = np.where(peaks_mask)
    points = list(zip(xs.tolist(), ys.tolist()))  # (x,y)

    if len(points) <= 1:
        return points

    # non-maximum suppression / distance-based greedy filtering
    kept = []
    points_sorted = sorted(points, key=lambda p: den_s[p[1], p[0]], reverse=True)
    for p in points_sorted:
        too_close = False
        for q in kept:
            if (p[0]-q[0])**2 + (p[1]-q[1])**2 <= (min_distance**2):
                too_close = True
                break
        if not too_close:
            kept.append(p)
    return kept
