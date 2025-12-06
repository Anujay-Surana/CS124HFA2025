# heatmap_visualizer.py
import cv2
import numpy as np

class HeatmapVisualizer:
    """
    Generates and overlays heat maps on video frames.
    """

    def __init__(self, alpha=0.5):
        """
        Initialize heatmap visualizer.

        Args:
            alpha: Transparency for heatmap overlay (0=transparent, 1=opaque)
        """
        self.alpha = alpha

    def apply_colormap(self, heatmap, colormap=cv2.COLORMAP_JET):
        """
        Apply color map to normalized heatmap.

        Args:
            heatmap: 2D numpy array (normalized 0-1)
            colormap: OpenCV colormap constant

        Returns:
            RGB image with applied colormap
        """
        # Convert to 8-bit
        heatmap_8bit = (heatmap * 255).astype(np.uint8)

        # Apply Gaussian blur for smoother visualization
        heatmap_8bit = cv2.GaussianBlur(heatmap_8bit, (15, 15), 0)

        # Apply colormap
        colored_heatmap = cv2.applyColorMap(heatmap_8bit, colormap)

        return colored_heatmap

    def resize_heatmap(self, heatmap, target_size):
        """
        Resize heatmap to match frame size.

        Args:
            heatmap: 2D numpy array
            target_size: (width, height) tuple

        Returns:
            Resized heatmap
        """
        return cv2.resize(heatmap, target_size, interpolation=cv2.INTER_LINEAR)

    def overlay_heatmap(self, frame, heatmap, alpha=None, colormap=cv2.COLORMAP_JET):
        """
        Overlay heatmap on frame.

        Args:
            frame: BGR frame
            heatmap: 2D numpy array (normalized 0-1)
            alpha: Override default transparency
            colormap: OpenCV colormap

        Returns:
            Frame with heatmap overlay
        """
        if alpha is None:
            alpha = self.alpha

        h, w = frame.shape[:2]

        # Resize heatmap to match frame
        heatmap_resized = self.resize_heatmap(heatmap, (w, h))

        # Apply colormap
        colored_heatmap = self.apply_colormap(heatmap_resized, colormap)

        # Create mask to only show non-zero areas
        mask = (heatmap_resized > 0.01).astype(np.uint8) * 255
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # Blend with frame
        overlay = cv2.addWeighted(frame, 1 - alpha, colored_heatmap, alpha, 0)

        # Use mask to keep original frame where there's no heat
        result = np.where(mask > 0, overlay, frame)

        return result.astype(np.uint8)

    def draw_trajectory(self, frame, trajectory, color=(0, 255, 255), thickness=2):
        """
        Draw person trajectory on frame.

        Args:
            frame: BGR frame
            trajectory: List of (x, y) points
            color: Line color (BGR)
            thickness: Line thickness

        Returns:
            Frame with trajectory drawn
        """
        if len(trajectory) < 2:
            return frame

        # Draw lines between consecutive points
        for i in range(1, len(trajectory)):
            pt1 = trajectory[i - 1]
            pt2 = trajectory[i]
            cv2.line(frame, pt1, pt2, color, thickness)

        # Draw dots at each point
        for pt in trajectory:
            cv2.circle(frame, pt, 3, color, -1)

        return frame

    def create_legend(self, height=50, width=300):
        """
        Create a color legend for the heatmap.

        Args:
            height: Height of legend bar
            width: Width of legend bar

        Returns:
            Legend image
        """
        # Create gradient
        gradient = np.linspace(0, 255, width, dtype=np.uint8)
        gradient = np.tile(gradient, (height, 1))

        # Apply colormap
        legend = cv2.applyColorMap(gradient, cv2.COLORMAP_JET)

        # Add text labels
        cv2.putText(legend, "Low", (10, height - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(legend, "High", (width - 50, height - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return legend

    def add_legend_to_frame(self, frame, legend_position="bottom-right"):
        """
        Add heatmap legend to frame.

        Args:
            frame: BGR frame
            legend_position: "bottom-right", "bottom-left", "top-right", "top-left"

        Returns:
            Frame with legend
        """
        legend = self.create_legend()
        lh, lw = legend.shape[:2]
        fh, fw = frame.shape[:2]

        # Determine position
        if legend_position == "bottom-right":
            y_offset = fh - lh - 10
            x_offset = fw - lw - 10
        elif legend_position == "bottom-left":
            y_offset = fh - lh - 10
            x_offset = 10
        elif legend_position == "top-right":
            y_offset = 10
            x_offset = fw - lw - 10
        else:  # top-left
            y_offset = 10
            x_offset = 10

        # Ensure within bounds
        y_offset = max(0, min(y_offset, fh - lh))
        x_offset = max(0, min(x_offset, fw - lw))

        # Overlay legend
        frame[y_offset:y_offset + lh, x_offset:x_offset + lw] = legend

        return frame

    def create_side_by_side(self, frame1, frame2, labels=None):
        """
        Create side-by-side comparison.

        Args:
            frame1: First frame
            frame2: Second frame
            labels: Optional tuple of (label1, label2)

        Returns:
            Combined frame
        """
        # Resize to same height
        h1, w1 = frame1.shape[:2]
        h2, w2 = frame2.shape[:2]
        target_h = max(h1, h2)

        frame1_resized = cv2.resize(frame1, (int(w1 * target_h / h1), target_h))
        frame2_resized = cv2.resize(frame2, (int(w2 * target_h / h2), target_h))

        # Add labels if provided
        if labels:
            cv2.putText(frame1_resized, labels[0], (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame2_resized, labels[1], (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Concatenate horizontally
        combined = np.hstack((frame1_resized, frame2_resized))

        return combined

    def draw_grid(self, frame, grid_size, color=(100, 100, 100), thickness=1):
        """
        Draw grid overlay on frame.

        Args:
            frame: BGR frame
            grid_size: Size of grid cells
            color: Grid line color
            thickness: Line thickness

        Returns:
            Frame with grid overlay
        """
        h, w = frame.shape[:2]

        # Draw vertical lines
        for x in range(0, w, grid_size):
            cv2.line(frame, (x, 0), (x, h), color, thickness)

        # Draw horizontal lines
        for y in range(0, h, grid_size):
            cv2.line(frame, (0, y), (w, y), color, thickness)

        return frame

    def draw_density_values(self, frame, density_map, grid_size, threshold=1):
        """
        Draw density values on grid cells.

        Args:
            frame: BGR frame
            density_map: 2D array of density values
            grid_size: Size of grid cells
            threshold: Minimum value to display

        Returns:
            Frame with density values
        """
        rows, cols = density_map.shape

        for row in range(rows):
            for col in range(cols):
                value = density_map[row, col]
                if value >= threshold:
                    # Calculate center of cell
                    cx = col * grid_size + grid_size // 2
                    cy = row * grid_size + grid_size // 2

                    # Draw value
                    text = str(int(value))
                    cv2.putText(frame, text, (cx - 10, cy + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return frame
