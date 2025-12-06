# face_heatmap_integration.py
"""
Heatmap integration for face tracking system.
Generates density heatmaps and trajectory visualizations from tracked faces.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
import json
from datetime import datetime


class FaceTrackingHeatmap:
    def __init__(self, width, height, sigma=40, enable_per_person=True):
        """
        Initialize heatmap generator for face tracking system.
        
        Args:
            width: Frame width in pixels
            height: Frame height in pixels
            sigma: Gaussian blur sigma for smoothing
            enable_per_person: Track individual person heatmaps
        """
        self.width = width
        self.height = height
        self.sigma = sigma
        self.enable_per_person = enable_per_person
        
        # Global heatmap (all faces)
        self.global_heatmap = np.zeros((height, width), dtype=np.float32)
        
        # Per-person heatmaps
        self.person_heatmaps = {}
        self.person_trajectories = {}
        
        # Frame tracking
        self.frame_count = 0
        self.total_detections = 0
        
        # Statistics
        self.person_stats = {}  # Track time spent, area coverage per person
        
    def update(self, bboxes, visible_dict):
        """
        Update heatmaps from current frame's tracking data.
        Compatible with CentroidTracker output format.
        
        Args:
            bboxes: Dict of {person_id: (x, y, w, h)} from ct.bboxes
            visible_dict: Dict of {person_id: bool} from ct.visible
        """
        self.frame_count += 1
        
        for person_id, bbox in bboxes.items():
            # Skip if person is not visible
            if not visible_dict.get(person_id, False):
                continue
            
            if bbox is None:
                continue
                
            x, y, w, h = bbox
            center_x = x + w // 2
            center_y = y + h // 2
            
            # Validate coordinates
            if not (0 <= center_x < self.width and 0 <= center_y < self.height):
                continue
            
            self.total_detections += 1
            
            # Update global heatmap
            self._add_to_heatmap(self.global_heatmap, center_x, center_y, w, h)
            
            # Update per-person data
            if self.enable_per_person:
                if person_id not in self.person_heatmaps:
                    self.person_heatmaps[person_id] = np.zeros((self.height, self.width), dtype=np.float32)
                    self.person_trajectories[person_id] = []
                    self.person_stats[person_id] = {
                        'first_seen': self.frame_count,
                        'last_seen': self.frame_count,
                        'total_frames': 0,
                        'avg_face_size': 0
                    }
                
                self._add_to_heatmap(self.person_heatmaps[person_id], center_x, center_y, w, h)
                self.person_trajectories[person_id].append((center_x, center_y, self.frame_count))
                
                # Update stats
                stats = self.person_stats[person_id]
                stats['last_seen'] = self.frame_count
                stats['total_frames'] += 1
                stats['avg_face_size'] = (stats['avg_face_size'] * (stats['total_frames'] - 1) + w * h) / stats['total_frames']
    
    def _add_to_heatmap(self, heatmap, center_x, center_y, width, height):
        """Add a face detection to a heatmap with face-size weighting."""
        # Weight by face size (larger faces = more weight, closer to camera)
        area = width * height
        weight = np.sqrt(area) / 100.0  # Normalize by typical face size
        weight = max(0.5, min(weight, 2.0))  # Clamp between 0.5 and 2.0
        
        # Add to heatmap
        cy, cx = int(center_y), int(center_x)
        if 0 <= cy < self.height and 0 <= cx < self.width:
            heatmap[cy, cx] += weight
    
    def generate_global_heatmap(self, normalize=True):
        """Generate smoothed global heatmap."""
        smoothed = gaussian_filter(self.global_heatmap, sigma=self.sigma)
        
        if normalize and smoothed.max() > 0:
            smoothed = smoothed / smoothed.max()
        
        return smoothed
    
    def generate_person_heatmap(self, person_id, normalize=True):
        """Generate smoothed heatmap for a specific person."""
        if person_id not in self.person_heatmaps:
            return None
        
        smoothed = gaussian_filter(self.person_heatmaps[person_id], sigma=self.sigma)
        
        if normalize and smoothed.max() > 0:
            smoothed = smoothed / smoothed.max()
        
        return smoothed
    
    def visualize_global_heatmap(self, save_path=None, background_frame=None, 
                                 colormap='hot', alpha=0.7, title=None, show_background=True):
        """
        Visualize the global heatmap.
        
        Args:
            save_path: Path to save the figure
            background_frame: Optional background frame to overlay
            colormap: Matplotlib colormap
            alpha: Transparency of heatmap (ignored if show_background=False)
            title: Custom title for the plot
            show_background: If False, shows heatmap on black background with no transparency
        """
        heatmap = self.generate_global_heatmap()
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Background frame
        if show_background and background_frame is not None:
            if len(background_frame.shape) == 3:
                background_frame = cv2.cvtColor(background_frame, cv2.COLOR_BGR2RGB)
            ax.imshow(background_frame, aspect='auto', extent=[0, self.width, self.height, 0])
            heatmap_alpha = alpha
        else:
            # Black background, no transparency
            ax.set_facecolor('black')
            fig.patch.set_facecolor('black')
            heatmap_alpha = 1.0
        
        # Heatmap overlay
        im = ax.imshow(heatmap, cmap=colormap, alpha=heatmap_alpha, 
                      extent=[0, self.width, self.height, 0],
                      interpolation='bilinear')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Face Density', fontsize=12)
        
        # Labels and title - white if no background
        label_color = 'white' if not (show_background and background_frame is not None) else 'black'
        ax.set_xlabel('X Position (pixels)', fontsize=12, color=label_color)
        ax.set_ylabel('Y Position (pixels)', fontsize=12, color=label_color)
        ax.tick_params(colors=label_color)
        cbar.ax.yaxis.set_tick_params(color=label_color)
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=label_color)
        cbar.set_label('Face Density', fontsize=12, color=label_color)
        
        if title is None:
            title = f'Global Face Tracking Heatmap\n{self.total_detections} detections across {self.frame_count} frames'
        ax.set_title(title, fontsize=14, fontweight='bold', color=label_color)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
            print(f"✓ Global heatmap saved → {save_path}")
        
        plt.close(fig)
        return fig
    
    def visualize_person_heatmap(self, person_id, save_path=None, background_frame=None,
                                colormap='plasma', alpha=0.7, show_background=True):
        """Visualize heatmap for a specific person."""
        heatmap = self.generate_person_heatmap(person_id)
        
        if heatmap is None:
            print(f"No data for person {person_id}")
            return None
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        if show_background and background_frame is not None:
            if len(background_frame.shape) == 3:
                background_frame = cv2.cvtColor(background_frame, cv2.COLOR_BGR2RGB)
            ax.imshow(background_frame, aspect='auto', extent=[0, self.width, self.height, 0])
            heatmap_alpha = alpha
        else:
            ax.set_facecolor('black')
            fig.patch.set_facecolor('black')
            heatmap_alpha = 1.0
        
        im = ax.imshow(heatmap, cmap=colormap, alpha=heatmap_alpha,
                      extent=[0, self.width, self.height, 0],
                      interpolation='bilinear')
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        label_color = 'white' if not (show_background and background_frame is not None) else 'black'
        ax.set_xlabel('X Position (pixels)', fontsize=12, color=label_color)
        ax.set_ylabel('Y Position (pixels)', fontsize=12, color=label_color)
        ax.tick_params(colors=label_color)
        cbar.ax.yaxis.set_tick_params(color=label_color)
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=label_color)
        cbar.set_label('Density', fontsize=12, color=label_color)
        
        # Get stats
        stats = self.person_stats.get(person_id, {})
        detections = stats.get('total_frames', 0)
        
        ax.set_title(f'Person {person_id} Movement Heatmap\n{detections} detections',
                    fontsize=14, fontweight='bold', color=label_color)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
            print(f"✓ Person {person_id} heatmap saved → {save_path}")
        
        plt.close(fig)
        return fig
    
    def visualize_person_trajectory(self, person_id, save_path=None, background_frame=None,
                                   max_points=2000, show_time_gradient=True, show_background=True):
        """Visualize movement trajectory for a specific person."""
        if person_id not in self.person_trajectories:
            print(f"No trajectory data for person {person_id}")
            return None
        
        trajectory = self.person_trajectories[person_id]
        if len(trajectory) == 0:
            print(f"Empty trajectory for person {person_id}")
            return None
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Background
        if show_background and background_frame is not None:
            if len(background_frame.shape) == 3:
                background_frame = cv2.cvtColor(background_frame, cv2.COLOR_BGR2RGB)
            ax.imshow(background_frame, aspect='auto', extent=[0, self.width, self.height, 0], alpha=0.6)
            bg_color = False
        else:
            ax.set_facecolor('black')
            fig.patch.set_facecolor('black')
            bg_color = True
        
        # Sample trajectory if too many points
        if len(trajectory) > max_points:
            indices = np.linspace(0, len(trajectory)-1, max_points, dtype=int)
            trajectory = [trajectory[i] for i in indices]
        
        x_coords = [pos[0] for pos in trajectory]
        y_coords = [pos[1] for pos in trajectory]
        
        if show_time_gradient:
            # Color by time progression
            colors = np.linspace(0, 1, len(x_coords))
            scatter = ax.scatter(x_coords, y_coords, c=colors, cmap='viridis', 
                               alpha=0.7, s=20, edgecolors='white', linewidths=0.5)
            cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
            if bg_color:
                cbar.ax.yaxis.set_tick_params(color='white')
                plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
                cbar.set_label('Time Progression', fontsize=12, color='white')
            else:
                cbar.set_label('Time Progression', fontsize=12)
        else:
            ax.scatter(x_coords, y_coords, c='red', alpha=0.6, s=20)
        
        # Draw connecting lines
        line_color = 'cyan' if bg_color else 'blue'
        ax.plot(x_coords, y_coords, line_color, alpha=0.6, linewidth=1.5)
        
        # Mark start and end
        ax.scatter(x_coords[0], y_coords[0], c='green', s=200, marker='o', 
                  edgecolors='white', linewidths=2, label='Start', zorder=5)
        ax.scatter(x_coords[-1], y_coords[-1], c='red', s=200, marker='X', 
                  edgecolors='white', linewidths=2, label='End', zorder=5)
        
        ax.set_xlim(0, self.width)
        ax.set_ylim(self.height, 0)
        
        label_color = 'white' if bg_color else 'black'
        ax.set_xlabel('X Position (pixels)', fontsize=12, color=label_color)
        ax.set_ylabel('Y Position (pixels)', fontsize=12, color=label_color)
        ax.set_title(f'Person {person_id} Movement Trajectory\n{len(self.person_trajectories[person_id])} positions tracked',
                    fontsize=14, fontweight='bold', color=label_color)
        ax.tick_params(colors=label_color)
        
        legend = ax.legend(loc='upper right')
        if bg_color:
            legend.get_frame().set_facecolor('black')
            legend.get_frame().set_edgecolor('white')
            for text in legend.get_texts():
                text.set_color('white')
        
        if bg_color:
            ax.grid(True, alpha=0.2, color='white')
        else:
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
            print(f"✓ Person {person_id} trajectory saved → {save_path}")
        
        plt.close(fig)
        return fig
        
        plt.close(fig)
        return fig
    
    def visualize_all_trajectories(self, save_path=None, background_frame=None, max_persons=10, show_background=True):
        """Visualize trajectories for all tracked persons on one plot."""
        if not self.person_trajectories:
            print("No trajectory data available")
            return None
        
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Background
        if show_background and background_frame is not None:
            if len(background_frame.shape) == 3:
                background_frame = cv2.cvtColor(background_frame, cv2.COLOR_BGR2RGB)
            ax.imshow(background_frame, aspect='auto', extent=[0, self.width, self.height, 0], alpha=0.5)
            bg_color = False
        else:
            ax.set_facecolor('black')
            fig.patch.set_facecolor('black')
            bg_color = True
        
        # Color palette
        colors = plt.cm.tab10(np.linspace(0, 1, min(len(self.person_trajectories), 10)))
        
        # Plot each person's trajectory
        for idx, (person_id, trajectory) in enumerate(sorted(self.person_trajectories.items())[:max_persons]):
            if len(trajectory) == 0:
                continue
            
            x_coords = [pos[0] for pos in trajectory]
            y_coords = [pos[1] for pos in trajectory]
            
            color = colors[idx % len(colors)]
            
            # Plot trajectory
            ax.plot(x_coords, y_coords, '-', color=color, alpha=0.8, linewidth=2, label=f'Person {person_id}')
            ax.scatter(x_coords[0], y_coords[0], c=[color], s=100, marker='o', edgecolors='white', linewidths=1.5, zorder=5)
            ax.scatter(x_coords[-1], y_coords[-1], c=[color], s=100, marker='X', edgecolors='white', linewidths=1.5, zorder=5)
        
        ax.set_xlim(0, self.width)
        ax.set_ylim(self.height, 0)
        
        label_color = 'white' if bg_color else 'black'
        ax.set_xlabel('X Position (pixels)', fontsize=12, color=label_color)
        ax.set_ylabel('Y Position (pixels)', fontsize=12, color=label_color)
        ax.set_title(f'All Person Trajectories (Showing {min(len(self.person_trajectories), max_persons)} persons)',
                    fontsize=14, fontweight='bold', color=label_color)
        ax.tick_params(colors=label_color)
        
        legend = ax.legend(loc='upper right', fontsize=10)
        if bg_color:
            legend.get_frame().set_facecolor('black')
            legend.get_frame().set_edgecolor('white')
            for text in legend.get_texts():
                text.set_color('white')
        
        if bg_color:
            ax.grid(True, alpha=0.2, color='white')
        else:
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
            print(f"✓ All trajectories saved → {save_path}")
        
        plt.close(fig)
        return fig
    
    def create_combined_visualization(self, person_id=None, save_path=None, background_frame=None, show_background=True):
        """Create a combined visualization with heatmap and trajectory side by side."""
        if person_id is None:
            # Global view
            heatmap = self.generate_global_heatmap()
            title_prefix = "Global"
        else:
            # Per-person view
            heatmap = self.generate_person_heatmap(person_id)
            if heatmap is None:
                print(f"No data for person {person_id}")
                return None
            title_prefix = f"Person {person_id}"
        
        fig, axes = plt.subplots(1, 2, figsize=(20, 9))
        
        # Determine if using background
        use_bg = show_background and background_frame is not None
        
        # Set figure background
        if not use_bg:
            fig.patch.set_facecolor('black')
            for ax in axes:
                ax.set_facecolor('black')
        
        # === Heatmap (left) ===
        if use_bg:
            bg = cv2.cvtColor(background_frame, cv2.COLOR_BGR2RGB) if len(background_frame.shape) == 3 else background_frame
            axes[0].imshow(bg, aspect='auto', extent=[0, self.width, self.height, 0])
            heatmap_alpha = 0.7
        else:
            heatmap_alpha = 1.0
        
        im = axes[0].imshow(heatmap, cmap='hot', alpha=heatmap_alpha,
                           extent=[0, self.width, self.height, 0],
                           interpolation='bilinear')
        cbar = plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
        
        # Color settings for heatmap
        label_color = 'white' if not use_bg else 'black'
        cbar.set_label('Density', fontsize=11, color=label_color)
        cbar.ax.yaxis.set_tick_params(color=label_color)
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=label_color)
        
        axes[0].set_xlabel('X Position', fontsize=11, color=label_color)
        axes[0].set_ylabel('Y Position', fontsize=11, color=label_color)
        axes[0].set_title(f'{title_prefix} - Density Heatmap', fontsize=13, fontweight='bold', color=label_color)
        axes[0].tick_params(colors=label_color)
        
        # === Trajectory (right) ===
        if use_bg:
            axes[1].imshow(bg, aspect='auto', extent=[0, self.width, self.height, 0], alpha=0.6)
        
        if person_id is None:
            # Show all trajectories
            colors = plt.cm.tab10(np.linspace(0, 1, min(len(self.person_trajectories), 10)))
            for idx, (pid, trajectory) in enumerate(sorted(self.person_trajectories.items())[:10]):
                if len(trajectory) == 0:
                    continue
                x_coords = [pos[0] for pos in trajectory]
                y_coords = [pos[1] for pos in trajectory]
                color = colors[idx % len(colors)]
                axes[1].plot(x_coords, y_coords, '-', color=color, alpha=0.8, linewidth=1.5, label=f'P{pid}')
            
            legend = axes[1].legend(loc='upper right', fontsize=9)
            if not use_bg:
                legend.get_frame().set_facecolor('black')
                legend.get_frame().set_edgecolor('white')
                for text in legend.get_texts():
                    text.set_color('white')
        else:
            # Show single person trajectory
            if person_id in self.person_trajectories:
                trajectory = self.person_trajectories[person_id]
                x_coords = [pos[0] for pos in trajectory]
                y_coords = [pos[1] for pos in trajectory]
                colors_grad = np.linspace(0, 1, len(x_coords))
                axes[1].scatter(x_coords, y_coords, c=colors_grad, cmap='viridis', 
                              alpha=0.7, s=20, edgecolors='white', linewidths=0.5)
                
                line_color = 'cyan' if not use_bg else 'blue'
                axes[1].plot(x_coords, y_coords, line_color, alpha=0.6, linewidth=1.5)
                axes[1].scatter(x_coords[0], y_coords[0], c='green', s=150, marker='o', 
                              edgecolors='white', linewidths=2, label='Start', zorder=5)
                axes[1].scatter(x_coords[-1], y_coords[-1], c='red', s=150, marker='X', 
                              edgecolors='white', linewidths=2, label='End', zorder=5)
                
                legend = axes[1].legend(loc='upper right')
                if not use_bg:
                    legend.get_frame().set_facecolor('black')
                    legend.get_frame().set_edgecolor('white')
                    for text in legend.get_texts():
                        text.set_color('white')
        
        axes[1].set_xlim(0, self.width)
        axes[1].set_ylim(self.height, 0)
        axes[1].set_xlabel('X Position', fontsize=11, color=label_color)
        axes[1].set_ylabel('Y Position', fontsize=11, color=label_color)
        axes[1].set_title(f'{title_prefix} - Movement Trajectory', fontsize=13, fontweight='bold', color=label_color)
        axes[1].tick_params(colors=label_color)
        
        if not use_bg:
            axes[1].grid(True, alpha=0.2, color='white')
        else:
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
            print(f"✓ Combined visualization saved → {save_path}")
        
        plt.close(fig)
        return fig
    
    def generate_summary_report(self, save_path=None):
        """Generate a text summary report of tracking statistics."""
        report = []
        report.append("=" * 70)
        report.append("FACE TRACKING HEATMAP ANALYSIS REPORT")
        report.append("=" * 70)
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"\nTotal Frames Processed: {self.frame_count}")
        report.append(f"Total Face Detections: {self.total_detections}")
        report.append(f"Unique Persons Tracked: {len(self.person_heatmaps)}")
        
        if self.frame_count > 0:
            report.append(f"Average Detections per Frame: {self.total_detections / self.frame_count:.2f}")
        
        report.append("\n" + "-" * 70)
        report.append("PER-PERSON STATISTICS")
        report.append("-" * 70)
        
        for person_id in sorted(self.person_heatmaps.keys()):
            stats = self.person_stats.get(person_id, {})
            report.append(f"\nPerson {person_id}:")
            report.append(f"  First Seen: Frame {stats.get('first_seen', 'N/A')}")
            report.append(f"  Last Seen: Frame {stats.get('last_seen', 'N/A')}")
            report.append(f"  Total Frames Visible: {stats.get('total_frames', 0)}")
            
            if self.frame_count > 0:
                visibility = (stats.get('total_frames', 0) / self.frame_count) * 100
                report.append(f"  Visibility: {visibility:.1f}%")
            
            avg_size = stats.get('avg_face_size', 0)
            report.append(f"  Average Face Size: {avg_size:.0f} px²")
            
            if person_id in self.person_trajectories:
                trajectory = self.person_trajectories[person_id]
                if len(trajectory) > 1:
                    # Calculate total distance traveled
                    total_dist = 0
                    for i in range(1, len(trajectory)):
                        x1, y1, _ = trajectory[i-1]
                        x2, y2, _ = trajectory[i]
                        dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                        total_dist += dist
                    report.append(f"  Total Distance Traveled: {total_dist:.0f} px")
        
        report.append("\n" + "=" * 70)
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"✓ Summary report saved → {save_path}")
        
        return report_text
    
    def export_heatmap_data(self, save_dir):
        """Export heatmap data as numpy arrays for further analysis."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save global heatmap
        np.save(os.path.join(save_dir, 'global_heatmap.npy'), self.global_heatmap)
        
        # Save per-person heatmaps
        for person_id, heatmap in self.person_heatmaps.items():
            np.save(os.path.join(save_dir, f'person_{person_id}_heatmap.npy'), heatmap)
        
        # Save trajectories as JSON (convert NumPy types to Python types)
        trajectories_json = {}
        for person_id, trajectory in self.person_trajectories.items():
            # Convert each (x, y, frame) tuple to Python ints
            trajectories_json[str(person_id)] = [
                (int(x), int(y), int(frame)) for x, y, frame in trajectory
            ]
        
        with open(os.path.join(save_dir, 'trajectories.json'), 'w') as f:
            json.dump(trajectories_json, f, indent=2)
        
        print(f"✓ Heatmap data exported → {save_dir}")
    
    def reset(self):
        """Reset all heatmap data."""
        self.global_heatmap = np.zeros((self.height, self.width), dtype=np.float32)
        self.person_heatmaps = {}
        self.person_trajectories = {}
        self.person_stats = {}
        self.frame_count = 0
        self.total_detections = 0
