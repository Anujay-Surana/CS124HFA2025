# centroid_tracker.py
from collections import OrderedDict
import numpy as np

class CentroidTracker:
    def __init__(self, max_disappeared=30, max_distance=100, min_feature_similarity=0.5, known_persons=None):
        """
        Initialize the centroid tracker with feature-based matching.

        Args:
            max_disappeared: Maximum frames an object can be lost before deregistration
            max_distance: Maximum pixel distance for position-based matching
            min_feature_similarity: Minimum cosine similarity (0-1) for feature matching
            known_persons: Dict of {person_id: feature_vector} to pre-register known persons
        """
        # Initialize counters and dictionaries
        self.next_object_id = 0
        self.objects = OrderedDict()  # Store centroids
        self.bboxes = OrderedDict()   # Store bounding boxes
        self.features = OrderedDict() # Store current face feature vectors (can drift)
        self.reference_features = OrderedDict() # Store reference features (never updated)
        self.disappeared = OrderedDict()
        self.visible = OrderedDict()  # Track if object is visible in current frame
        self.similarities = OrderedDict()  # Store similarity scores for display

        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.min_feature_similarity = min_feature_similarity

        # Pre-register known persons from previous sessions
        if known_persons:
            for pid, feature in known_persons.items():
                self.features[pid] = feature
                self.reference_features[pid] = feature.copy()  # Save reference
                self.objects[pid] = None  # Will be set when first detected
                self.bboxes[pid] = None
                self.disappeared[pid] = 0
                self.visible[pid] = False
                self.similarities[pid] = 0.0
                # Update next_object_id to avoid conflicts
                if pid >= self.next_object_id:
                    self.next_object_id = pid + 1

    def register(self, centroid, bbox, feature=None):
        """
        Register a new tracked object.

        Args:
            centroid: (x, y) center point
            bbox: (x, y, w, h) bounding box
            feature: 128-dim face feature vector (optional)
        """
        self.objects[self.next_object_id] = centroid
        self.bboxes[self.next_object_id] = bbox
        self.features[self.next_object_id] = feature
        self.reference_features[self.next_object_id] = feature.copy() if feature is not None else None
        self.disappeared[self.next_object_id] = 0
        self.visible[self.next_object_id] = True
        self.similarities[self.next_object_id] = 0.0  # New person starts with 0
        self.next_object_id += 1

    def deregister(self, object_id):
        """Remove an object that has disappeared for too long."""
        del self.objects[object_id]
        del self.bboxes[object_id]
        del self.features[object_id]
        del self.reference_features[object_id]
        del self.disappeared[object_id]
        del self.visible[object_id]
        del self.similarities[object_id]

    def update(self, rects, features=None):
        """
        Update tracked objects with new detections.

        Args:
            rects: List of (x, y, w, h) bounding boxes
            features: List of 128-dim feature vectors (same length as rects)

        Returns:
            objects: Dict of {object_id: centroid}
            bboxes: Dict of {object_id: bbox}
        """
        print(f"\n[TRACKER] Update called with {len(rects)} detection(s)")

        # Mark all existing objects as not visible initially
        for object_id in self.visible.keys():
            self.visible[object_id] = False

        # Handle no detections
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                # Don't deregister - keep all persons in history for future matching
            return self.objects, self.bboxes

        # Compute centroids for new detections
        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (x, y, w, h)) in enumerate(rects):
            input_centroids[i] = (int(x + w / 2), int(y + h / 2))

        # If no existing objects, register all new detections
        if len(self.objects) == 0:
            for i in range(len(rects)):
                feat = features[i] if features is not None else None
                self.register(input_centroids[i], rects[i], feat)
            return self.objects, self.bboxes

        # Match existing objects with new detections
        # Separate into two categories:
        # 1. Recently seen (disappeared < 5 frames) - use position + feature
        # 2. Long disappeared or never seen - use feature only

        recently_seen_ids = [oid for oid in self.objects.keys()
                            if self.objects[oid] is not None and self.disappeared.get(oid, 0) < 5]
        recently_seen_centroids = [self.objects[oid] for oid in recently_seen_ids]

        long_disappeared_ids = [oid for oid in self.objects.keys()
                               if oid not in recently_seen_ids]

        # If no recently seen objects, all detections go to feature-only matching
        if len(recently_seen_ids) == 0:
            # Try to match all detections with long-disappeared persons
            for col in range(len(rects)):
                matched_inactive = False
                if features is not None and features[col] is not None:
                    # Try to match with long-disappeared persons using feature similarity
                    best_match_id = None
                    best_similarity = 0

                    for inactive_id in long_disappeared_ids:
                        if self.reference_features[inactive_id] is None:
                            continue

                        similarity = np.dot(self.reference_features[inactive_id], features[col]) / (
                            np.linalg.norm(self.reference_features[inactive_id]) * np.linalg.norm(features[col]) + 1e-8
                        )

                        if similarity > best_similarity and similarity > self.min_feature_similarity:
                            best_similarity = similarity
                            best_match_id = inactive_id

                    # Reactivate known person if match found
                    if best_match_id is not None:
                        print(f"[INACTIVE MATCH] Detection #{col} -> Person {best_match_id} (similarity: {best_similarity:.3f})")
                        self.objects[best_match_id] = input_centroids[col]
                        self.bboxes[best_match_id] = rects[col]
                        self.visible[best_match_id] = True
                        self.disappeared[best_match_id] = 0
                        self.similarities[best_match_id] = best_similarity  # Store similarity
                        # Update current feature with fast EMA
                        self.features[best_match_id] = 0.8 * self.features[best_match_id] + 0.2 * features[col]
                        self.features[best_match_id] = self.features[best_match_id] / (
                            np.linalg.norm(self.features[best_match_id]) + 1e-8
                        )
                        # Update reference feature with moderate EMA to adapt to pose changes
                        if best_similarity > 0.45:
                            self.reference_features[best_match_id] = 0.75 * self.reference_features[best_match_id] + 0.25 * features[col]
                            self.reference_features[best_match_id] = self.reference_features[best_match_id] / (
                                np.linalg.norm(self.reference_features[best_match_id]) + 1e-8
                            )
                        elif best_similarity > 0.25:
                            self.reference_features[best_match_id] = 0.88 * self.reference_features[best_match_id] + 0.12 * features[col]
                            self.reference_features[best_match_id] = self.reference_features[best_match_id] / (
                                np.linalg.norm(self.reference_features[best_match_id]) + 1e-8
                            )
                        matched_inactive = True
                        # Remove from long_disappeared list
                        long_disappeared_ids.remove(best_match_id)

                # Register as new if no match found
                if not matched_inactive:
                    print(f"[NEW PERSON] Detection #{col} registered as new Person {self.next_object_id}")
                    feat = features[col] if features is not None else None
                    self.register(input_centroids[col], rects[col], feat)

            return self.objects, self.bboxes

        # Compute position distance matrix for recently seen objects
        D_pos = np.linalg.norm(
            np.expand_dims(recently_seen_centroids, 1) - np.expand_dims(input_centroids, 0),
            axis=2
        )

        # Compute feature similarity matrix if features are provided
        if features is not None and any(f is not None for f in self.reference_features.values()):
            D_feat = np.zeros((len(recently_seen_ids), len(rects)))
            for i, obj_id in enumerate(recently_seen_ids):
                obj_feat = self.reference_features[obj_id]  # Use reference feature
                if obj_feat is None:
                    D_feat[i, :] = 0  # No feature, use position only
                    continue
                for j, new_feat in enumerate(features):
                    if new_feat is None:
                        D_feat[i, j] = 0
                        continue
                    # Cosine similarity (higher is better, range: -1 to 1)
                    similarity = np.dot(obj_feat, new_feat) / (
                        np.linalg.norm(obj_feat) * np.linalg.norm(new_feat) + 1e-8
                    )
                    D_feat[i, j] = similarity

            # Combined cost: normalize position distance and use (1 - similarity) for features
            # Lower is better for both metrics
            D_pos_norm = D_pos / (D_pos.max() + 1e-8)
            D_feat_cost = 1 - D_feat  # Convert similarity to cost
            D = 0.15 * D_pos_norm + 0.85 * D_feat_cost  # Heavily favor features over position
        else:
            D = D_pos  # Use position only if no features available

        # Match using minimum cost (greedy assignment)
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]
        used_rows, used_cols = set(), set()

        for (row, col) in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue

            # Check if match is valid using adaptive strategy
            pos_dist = D_pos[row, col]
            pos_valid = pos_dist < self.max_distance

            # Feature check
            feature_valid = False
            similarity = -1.0  # Default for logging
            if features is not None and self.reference_features[recently_seen_ids[row]] is not None:
                if features[col] is not None:
                    obj_feat = self.reference_features[recently_seen_ids[row]]  # Use reference feature
                    new_feat = features[col]
                    similarity = np.dot(obj_feat, new_feat) / (
                        np.linalg.norm(obj_feat) * np.linalg.norm(new_feat) + 1e-8
                    )
                    feature_valid = similarity > self.min_feature_similarity
            else:
                # If no features available, fall back to position only
                feature_valid = True

            # Simplified matching strategy: just check if feature is valid (similarity > threshold)
            # Since we're only matching recently seen objects (< 5 frames), position should be similar
            # But we trust features more, so if similarity is good, we match
            valid_match = feature_valid
            match_reason = "feature" if feature_valid else "none"

            # Log matching attempt
            object_id = recently_seen_ids[row]
            print(f"[MATCH] Detection #{col} -> Person {object_id}: pos_dist={pos_dist:.1f}, similarity={similarity:.3f} (valid={feature_valid}), match={valid_match} (reason={match_reason})")

            if not valid_match:
                continue

            # Update matched object
            object_id = recently_seen_ids[row]
            self.objects[object_id] = input_centroids[col]
            self.bboxes[object_id] = rects[col]
            self.visible[object_id] = True  # Mark as visible in current frame
            self.similarities[object_id] = similarity  # Store similarity score

            # Update feature if provided
            if features is not None and features[col] is not None:
                # Update current feature with fast EMA for short-term tracking
                if self.features[object_id] is not None:
                    self.features[object_id] = 0.8 * self.features[object_id] + 0.2 * features[col]
                    # Normalize to maintain unit vector properties
                    self.features[object_id] = self.features[object_id] / (
                        np.linalg.norm(self.features[object_id]) + 1e-8
                    )
                else:
                    self.features[object_id] = features[col]

                # Update reference feature with moderate EMA to adapt to pose changes
                if self.reference_features[object_id] is not None and similarity > 0.45:
                    # High confidence match - update faster
                    self.reference_features[object_id] = 0.75 * self.reference_features[object_id] + 0.25 * features[col]
                    self.reference_features[object_id] = self.reference_features[object_id] / (
                        np.linalg.norm(self.reference_features[object_id]) + 1e-8
                    )
                elif self.reference_features[object_id] is not None and similarity > 0.25:
                    # Medium confidence - update slower
                    self.reference_features[object_id] = 0.88 * self.reference_features[object_id] + 0.12 * features[col]
                    self.reference_features[object_id] = self.reference_features[object_id] / (
                        np.linalg.norm(self.reference_features[object_id]) + 1e-8
                    )

            self.disappeared[object_id] = 0
            used_rows.add(row)
            used_cols.add(col)

        # Mark unmatched recently seen objects as disappeared (but don't deregister)
        unused_rows = set(range(D.shape[0])) - used_rows
        for row in unused_rows:
            object_id = recently_seen_ids[row]
            self.disappeared[object_id] += 1
            # Don't deregister - keep in history for future matching

        # Check unmatched detections against long-disappeared persons
        unused_cols = set(range(D.shape[1])) - used_cols
        remaining_cols = set()

        for col in unused_cols:
            matched_inactive = False
            if features is not None and features[col] is not None:
                # Try to match with long-disappeared persons using feature similarity
                best_match_id = None
                best_similarity = 0

                for inactive_id in long_disappeared_ids:
                    if self.reference_features[inactive_id] is None:
                        continue

                    similarity = np.dot(self.reference_features[inactive_id], features[col]) / (
                        np.linalg.norm(self.reference_features[inactive_id]) * np.linalg.norm(features[col]) + 1e-8
                    )

                    if similarity > best_similarity and similarity > self.min_feature_similarity:
                        best_similarity = similarity
                        best_match_id = inactive_id

                # Reactivate known person if match found
                if best_match_id is not None:
                    print(f"[REACTIVATE] Detection #{col} -> Person {best_match_id} (similarity: {best_similarity:.3f})")
                    self.objects[best_match_id] = input_centroids[col]
                    self.bboxes[best_match_id] = rects[col]
                    self.visible[best_match_id] = True
                    self.disappeared[best_match_id] = 0
                    self.similarities[best_match_id] = best_similarity  # Store similarity
                    # Update current feature with fast EMA
                    self.features[best_match_id] = 0.8 * self.features[best_match_id] + 0.2 * features[col]
                    self.features[best_match_id] = self.features[best_match_id] / (
                        np.linalg.norm(self.features[best_match_id]) + 1e-8
                    )
                    # Update reference feature with moderate EMA to adapt to pose changes
                    if best_similarity > 0.45:
                        self.reference_features[best_match_id] = 0.75 * self.reference_features[best_match_id] + 0.25 * features[col]
                        self.reference_features[best_match_id] = self.reference_features[best_match_id] / (
                            np.linalg.norm(self.reference_features[best_match_id]) + 1e-8
                        )
                    elif best_similarity > 0.25:
                        self.reference_features[best_match_id] = 0.88 * self.reference_features[best_match_id] + 0.12 * features[col]
                        self.reference_features[best_match_id] = self.reference_features[best_match_id] / (
                            np.linalg.norm(self.reference_features[best_match_id]) + 1e-8
                        )
                    matched_inactive = True
                    # Remove from long_disappeared list
                    long_disappeared_ids.remove(best_match_id)

            if not matched_inactive:
                remaining_cols.add(col)
                print(f"[UNMATCHED] Detection #{col} will be registered as new person")

        # Register truly new detections as new objects
        for col in remaining_cols:
            feat = features[col] if features is not None else None
            print(f"[NEW PERSON] Detection #{col} registered as Person {self.next_object_id}")
            self.register(input_centroids[col], rects[col], feat)

        return self.objects, self.bboxes
