import cv2
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use Agg backend
import matplotlib.pyplot as plt
from PIL import Image
from rc_robot_inference_v2 import RCCarNet, RCCarPredictor
import os

class VideoPredictor:
    def __init__(self, model_path):
        self.predictor = RCCarPredictor(model_path)
        self.output_width = 1500
        self.output_height = 500
        
        # Load car icon with transparency
        self.car_icon = plt.imread('zaptrack.png')
        # If the image has 3 channels, convert to RGBA
        if self.car_icon.shape[-1] == 3:
            # Create alpha channel (make black pixels transparent)
            alpha = np.all(self.car_icon == [0, 0, 0], axis=-1)
            rgba = np.dstack((self.car_icon, ~alpha))
            self.car_icon = rgba
        
        # Set up style
        plt.style.use('dark_background')
        self.colors = {
            'background': '#1C1C1C',
            'text': '#FFFFFF',
            'accent': '#00FF00',  # Changed to green
            'grid': '#2C2C2C'
        }
        
    def create_visualization_frame(self, frame, steering, throttle):
        # The input frame is in BGR format; convert it to RGB format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        
        # Create figure with specific size to match output dimensions
        dpi = 100
        fig = plt.figure(figsize=(self.output_width/dpi, self.output_height/dpi), dpi=dpi)
        fig.patch.set_facecolor(self.colors['background'])
        
        gs = plt.GridSpec(1, 3, width_ratios=[1, 0.8, 0.8], wspace=0.3)
        
        # Plot input frame
        ax1 = fig.add_subplot(gs[0])
        ax1.imshow(frame_pil)
        ax1.axis('off')
        ax1.set_title('Input Frame', color=self.colors['text'], pad=10, fontsize=12)
        
        # Plot controls
        ax2 = fig.add_subplot(gs[1])
        ax2.set_facecolor(self.colors['background'])
        controls = {'Steering': steering, 'Throttle': throttle}
        bars = ax2.bar(controls.keys(), controls.values(), color=self.colors['accent'], alpha=0.7)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', color=self.colors['text'])
        
        ax2.axhline(y=1500, color='r', linestyle='--', alpha=0.5)
        ax2.set_ylim(1000, 2000)
        ax2.set_title('Predicted Controls', color=self.colors['text'], pad=10, fontsize=12)
        ax2.tick_params(colors=self.colors['text'])
        ax2.spines['bottom'].set_color(self.colors['grid'])
        ax2.spines['top'].set_color(self.colors['grid'])
        ax2.spines['left'].set_color(self.colors['grid'])
        ax2.spines['right'].set_color(self.colors['grid'])
        
        # Plot trajectory visualization
        ax3 = fig.add_subplot(gs[2])
        ax3.set_facecolor(self.colors['background'])
        self._plot_trajectory(ax3, steering, throttle)
        ax3.set_title('Predicted Trajectory', color=self.colors['text'], pad=10, fontsize=12)
        
        plt.tight_layout()
        
        # Convert plot to image
        fig.canvas.draw()
        
        # Get the RGBA buffer from the figure
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf.shape = (h, w, 4)
    
        vis_image = buf
        
        # Resize to desired output dimensions
        vis_image = cv2.resize(vis_image, (self.output_width, self.output_height))
        
        plt.close()
        return vis_image
    
    def _plot_trajectory(self, ax, steering, throttle):
        """Plot the predicted trajectory based on steering angle and throttle"""
        ax.set_xlim(-6, 6)
        ax.set_ylim(0, 12)
        ax.grid(True, alpha=0.2, color=self.colors['grid'])
        
        # Calculate steering angle in degrees (-15 to +15 degrees)
        steering_angle = (steering - 1500) / 500 * -30
        
        # Calculate trajectory points
        num_points = 50
        time_points = np.linspace(0, 3, num_points)  # Increased time range for longer trajectory
        
        # Initialize arrays for trajectory
        x_points = np.zeros(num_points)
        y_points = np.zeros(num_points)
        
        # Calculate vehicle speed based on throttle (normalized to 0-1 range)
        speed = max(0, (throttle - 1500) / 500)  # Only forward motion
        
        # Constants for trajectory calculation
        wheelbase = 1.2  # meters
        dt = time_points[1] - time_points[0]
        
        # Calculate trajectory using bicycle model
        x, y, theta = 0, 2.5, np.pi/2
        
        for i in range(num_points):
            # Store current position
            x_points[i] = x
            y_points[i] = y
            
            # Update position and orientation
            if abs(steering_angle) < 1:
                # Straight line motion
                dx = 0
                dy = speed * dt
            else:
                # Turning motion
                turn_radius = wheelbase / np.tan(np.radians(steering_angle))
                angular_velocity = speed / turn_radius
                
                dx = speed * np.cos(theta) * dt
                dy = speed * np.sin(theta) * dt
                theta += angular_velocity * dt
            
            x += dx
            y += dy
        
        # Plot trajectory path with gradient color and varying thickness
        points = np.array([x_points, y_points]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # Create line collection with varying colors and width
        from matplotlib.collections import LineCollection
        colors = np.linspace(0, 1, len(segments))
        lc = LineCollection(segments, color=self.colors['accent'],  # Use green color
                          linewidth=np.linspace(80, 32, len(segments)),  # Increased line width
                          alpha=np.linspace(0.8, 0.3, len(segments)))
        ax.add_collection(lc)
        
        # Plot car icon at start of trajectory
        car_x, car_y = x_points[0], y_points[0]
        car_width = 1.4  # Increased size
        car_height = 1.0  # Adjusted for aspect ratio
        
        # Calculate the exact center position for the car
        extent = [
            car_x - car_width/2,  # left
            car_x + car_width/2,  # right
            car_y - car_height/2,  # bottom
            car_y + car_height/2   # top
        ]
        
        # Create a new axes for the car icon
        car_ax = ax.inset_axes([
            0.5 - car_width/6,  # Center horizontally
            0.01,                # Position from bottom
            car_width/3,        # Width
            car_height/3        # Height
        ])
        
        # Display car icon with transparency
        car_ax.imshow(self.car_icon, extent=extent, aspect='auto')
        car_ax.axis('off')
        car_ax.set_zorder(10)  # Make sure car is drawn on top
        
        # Apply rotation around the car's center
        transform = matplotlib.transforms.Affine2D().rotate_deg_around(car_x, car_y, -steering_angle)
        car_ax.set_transform(car_ax.get_transform() + transform)
        
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Set background color
        ax.set_facecolor(self.colors['background'])
    
    def process_video(self, input_video_path, output_video_path):
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {input_video_path}")
            return
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Create output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (self.output_width, self.output_height))
        
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            print(f"\rProcessing frame {frame_count}/{total_frames} ({(frame_count/total_frames)*100:.1f}%)", end="")
            
            # Convert frame to PIL Image for prediction
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Get predictions
            steering, throttle = self.predictor.predict_image(frame_pil, visualize=False)
            
            # Create visualization frame
            vis_frame = self.create_visualization_frame(frame, steering, throttle)
            
            # Convert RGBA to BGR for video writing
            vis_frame_bgr = cv2.cvtColor(vis_frame, cv2.COLOR_RGBA2BGR)
            
            # Write frame
            out.write(vis_frame_bgr)
        
        print("\nVideo processing complete!")
        cap.release()
        out.release()

def main():
    # Set up paths
    model_path = 'best_rc_robot_model.pth'
    input_video = 'test_video/20240920_19_23.mp4'
    output_video = 'test_video/output_predictions.mp4'
    
    # Create video predictor and process video
    video_predictor = VideoPredictor(model_path)
    video_predictor.process_video(input_video, output_video)

if __name__ == "__main__":
    main() 