# centroid_tracker.py
from collections import OrderedDict
import numpy as np

class CentroidTracker:
    def __init__(self, max_disappeared=30):
        # Initialize counters and dictionaries
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.bboxes = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared

    def register(self, centroid, bbox):
        # Register a new object
        self.objects[self.next_object_id] = centroid
        self.bboxes[self.next_object_id] = bbox
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        # Remove object that disappeared too long
        del self.objects[object_id]
        del self.bboxes[object_id]
        del self.disappeared[object_id]

    def update(self, rects):
        # rects: list of (x, y, w, h)
        if len(rects) == 0:
            # Increment disappeared count for all objects
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects, self.bboxes

        # Compute centroids for the new detections
        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (x, y, w, h)) in enumerate(rects):
            input_centroids[i] = (int(x + w / 2), int(y + h / 2))

        # If no existing objects, register all
        if len(self.objects) == 0:
            for i in range(len(rects)):
                self.register(input_centroids[i], rects[i])
            return self.objects, self.bboxes

        # Compute distance between each existing object and new detections
        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())
        D = np.linalg.norm(
            np.expand_dims(object_centroids, 1) - np.expand_dims(input_centroids, 0),
            axis=2
        )

        # Match using minimum distance
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]
        used_rows, used_cols = set(), set()

        for (row, col) in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            object_id = object_ids[row]
            self.objects[object_id] = input_centroids[col]
            self.bboxes[object_id] = rects[col]
            self.disappeared[object_id] = 0
            used_rows.add(row)
            used_cols.add(col)

        # Mark unmatched existing objects
        unused_rows = set(range(D.shape[0])) - used_rows
        for row in unused_rows:
            object_id = object_ids[row]
            self.disappeared[object_id] += 1
            if self.disappeared[object_id] > self.max_disappeared:
                self.deregister(object_id)

        # Register unmatched new detections
        unused_cols = set(range(D.shape[1])) - used_cols
        for col in unused_cols:
            self.register(input_centroids[col], rects[col])

        return self.objects, self.bboxes