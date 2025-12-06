# analytics_tracker.py
import numpy as np
import time
from collections import defaultdict, deque
from datetime import datetime

class AnalyticsTracker:
    """
    Tracks analytics data for person detection and movement:
    - Movement patterns (position history)
    - Dwell time (time spent in frame)
    - Frequency (number of visits)
    - Density maps (heat zones)
    """

    def __init__(self, frame_width, frame_height, grid_size=20):
        """
        Initialize analytics tracker.

        Args:
            frame_width: Width of video frame
            frame_height: Height of video frame
            grid_size: Size of grid cells for density mapping
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.grid_size = grid_size

        # Calculate grid dimensions
        self.grid_cols = frame_width // grid_size
        self.grid_rows = frame_height // grid_size

        # Movement tracking
        self.position_history = defaultdict(lambda: deque(maxlen=100))  # Last 100 positions per person
        self.movement_heatmap = np.zeros((self.grid_rows, self.grid_cols), dtype=np.float32)

        # Dwell time tracking
        self.first_seen = {}  # {person_id: timestamp}
        self.last_seen = {}   # {person_id: timestamp}
        self.total_dwell_time = defaultdict(float)  # Total time in seconds
        self.current_session_start = {}  # Track current session start time

        # Frequency tracking
        self.visit_count = defaultdict(int)  # Number of times person appeared
        self.session_count = defaultdict(int)  # Number of distinct sessions

        # Density tracking
        self.density_heatmap = np.zeros((self.grid_rows, self.grid_cols), dtype=np.float32)
        self.instantaneous_density = np.zeros((self.grid_rows, self.grid_cols), dtype=np.int32)

        # Frame counter
        self.frame_count = 0

    def update(self, tracked_objects, visible_objects):
        """
        Update analytics with current frame data.

        Args:
            tracked_objects: Dict of {person_id: (cx, cy)} centroids
            visible_objects: Dict of {person_id: bool} visibility status
        """
        self.frame_count += 1
        current_time = time.time()

        # Reset instantaneous density for this frame
        self.instantaneous_density.fill(0)

        for person_id, centroid in tracked_objects.items():
            is_visible = visible_objects.get(person_id, False)

            if not is_visible:
                # Person not visible - end current session if active
                if person_id in self.current_session_start:
                    session_duration = current_time - self.current_session_start[person_id]
                    self.total_dwell_time[person_id] += session_duration
                    del self.current_session_start[person_id]
                continue

            cx, cy = centroid

            # Update position history
            self.position_history[person_id].append((cx, cy, current_time))

            # Update movement heatmap
            grid_x = min(int(cx / self.grid_size), self.grid_cols - 1)
            grid_y = min(int(cy / self.grid_size), self.grid_rows - 1)
            self.movement_heatmap[grid_y, grid_x] += 1

            # Update density heatmap (cumulative)
            self.density_heatmap[grid_y, grid_x] += 1

            # Update instantaneous density
            self.instantaneous_density[grid_y, grid_x] += 1

            # Update dwell time tracking
            if person_id not in self.first_seen:
                self.first_seen[person_id] = current_time
                self.visit_count[person_id] += 1

            self.last_seen[person_id] = current_time

            # Start or continue current session
            if person_id not in self.current_session_start:
                self.current_session_start[person_id] = current_time
                self.session_count[person_id] += 1

    def get_person_dwell_time(self, person_id):
        """Get total dwell time for a person in seconds."""
        total = self.total_dwell_time[person_id]

        # Add current active session time if person is still visible
        if person_id in self.current_session_start:
            total += time.time() - self.current_session_start[person_id]

        return total

    def get_movement_heatmap(self, normalize=True):
        """
        Get movement heatmap.

        Args:
            normalize: If True, normalize values to 0-1 range

        Returns:
            2D numpy array representing movement density
        """
        if normalize and self.movement_heatmap.max() > 0:
            return self.movement_heatmap / self.movement_heatmap.max()
        return self.movement_heatmap.copy()

    def get_density_heatmap(self, normalize=True):
        """
        Get cumulative density heatmap.

        Args:
            normalize: If True, normalize values to 0-1 range

        Returns:
            2D numpy array representing density
        """
        if normalize and self.density_heatmap.max() > 0:
            return self.density_heatmap / self.density_heatmap.max()
        return self.density_heatmap.copy()

    def get_instantaneous_density(self):
        """Get current frame density (number of people per grid cell)."""
        return self.instantaneous_density.copy()

    def get_person_trajectory(self, person_id, max_points=50):
        """
        Get recent trajectory for a person.

        Args:
            person_id: ID of person
            max_points: Maximum number of points to return

        Returns:
            List of (x, y) positions
        """
        history = list(self.position_history[person_id])
        if len(history) > max_points:
            # Sample evenly
            step = len(history) / max_points
            history = [history[int(i * step)] for i in range(max_points)]
        return [(int(x), int(y)) for x, y, _ in history]

    def get_analytics_summary(self):
        """
        Get summary of all analytics.

        Returns:
            Dictionary with analytics data
        """
        summary = {
            "total_frames": self.frame_count,
            "unique_persons": len(self.first_seen),
            "persons": []
        }

        for person_id in sorted(self.first_seen.keys()):
            dwell_time = self.get_person_dwell_time(person_id)
            person_data = {
                "id": int(person_id),
                "first_seen": datetime.fromtimestamp(self.first_seen[person_id]).strftime("%Y-%m-%d %H:%M:%S"),
                "last_seen": datetime.fromtimestamp(self.last_seen[person_id]).strftime("%Y-%m-%d %H:%M:%S"),
                "dwell_time_seconds": float(dwell_time),
                "visit_count": int(self.visit_count[person_id]),
                "session_count": int(self.session_count[person_id]),
                "trajectory_points": len(self.position_history[person_id])
            }
            summary["persons"].append(person_data)

        return summary

    def export_heatmaps(self, output_dir):
        """
        Export heatmap data to files.

        Args:
            output_dir: Directory to save heatmap files
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        # Save movement heatmap
        np.save(os.path.join(output_dir, "movement_heatmap.npy"), self.movement_heatmap)

        # Save density heatmap
        np.save(os.path.join(output_dir, "density_heatmap.npy"), self.density_heatmap)

        # Save analytics summary as JSON
        import json
        summary = self.get_analytics_summary()
        with open(os.path.join(output_dir, "analytics_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
