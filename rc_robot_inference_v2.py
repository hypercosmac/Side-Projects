import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
from pathlib import Path

class RCCarNet(nn.Module):
    def __init__(self):
        super(RCCarNet, self).__init__()
        # CNN layers
        self.conv1 = nn.Conv2d(3, 24, 5, stride=2)
        self.conv2 = nn.Conv2d(24, 36, 5, stride=2)
        self.conv3 = nn.Conv2d(36, 48, 5, stride=2)
        self.conv4 = nn.Conv2d(48, 64, 3)
        self.conv5 = nn.Conv2d(64, 64, 3)
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(24)
        self.bn2 = nn.BatchNorm2d(36)
        self.bn3 = nn.BatchNorm2d(48)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        
        # Fully connected layers
        self.fc1 = nn.Linear(1152, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, 2)
        
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        
        return x

def denormalize_rc_value(value):
    """Convert normalized [-1,1] value back to RC [1000-2000] range"""
    return (value * 500) + 1500

class RCCarPredictor:
    def __init__(self, model_path):
        # Initialize model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = RCCarNet().to(self.device)
        
        # Load trained weights
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Set up image transformation
        self.transform = transforms.Compose([
            transforms.Resize((66, 200)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
    
    def predict_image(self, image_input, visualize=True):
        """Predict steering and throttle for a single image
        Args:
            image_input: Can be either a file path or a PIL Image object
            visualize: Whether to show the visualization
        """
        # Handle both file path and PIL Image input
        if isinstance(image_input, str):
            image = Image.open(image_input)
        else:
            image = image_input
            
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            prediction = self.model(input_tensor)[0].cpu().numpy()
        
        # Denormalize predictions
        steering = denormalize_rc_value(prediction[0])
        throttle = denormalize_rc_value(prediction[1])
        
        if visualize:
            self.visualize_prediction(image, steering, throttle)
        
        return steering, throttle
    
    def visualize_prediction(self, image, steering, throttle):
        """Visualize the image with predicted controls and trajectory"""
        fig = plt.figure(figsize=(15, 5))
        gs = plt.GridSpec(1, 3, width_ratios=[1, 0.8, 0.8])
        
        # Plot input image
        ax1 = fig.add_subplot(gs[0])
        ax1.imshow(image)
        ax1.axis('off')
        ax1.set_title('Input Image')
        
        # Plot controls
        ax2 = fig.add_subplot(gs[1])
        controls = {'Steering': steering, 'Throttle': throttle}
        ax2.bar(controls.keys(), controls.values())
        ax2.axhline(y=1500, color='r', linestyle='--', alpha=0.5)
        ax2.set_ylim(1000, 2000)
        ax2.set_title('Predicted Controls')
        
        # Plot trajectory visualization
        ax3 = fig.add_subplot(gs[2])
        self._plot_trajectory(ax3, steering)
        ax3.set_title('Predicted Trajectory')
        
        plt.tight_layout()
        plt.show()
    
    def _plot_trajectory(self, ax, steering):
        """Plot the predicted trajectory based on steering angle"""
        # Create a simple top-down view
        ax.set_xlim(-2, 2)
        ax.set_ylim(0, 4)
        ax.grid(True, alpha=0.3)
        
        # Calculate steering angle in degrees (mapping from RC values)
        # 1000->-30°, 1500->0°, 2000->30°
        steering_angle = (steering - 1500) / 500 * 30  # degrees
        
        # Generate trajectory points
        t = np.linspace(0, 2*np.pi, 100)
        if abs(steering_angle) < 1:  # Nearly straight
            x = np.zeros_like(t)
            y = np.linspace(0, 3, len(t))
        else:
            # Calculate turning radius (smaller angle = larger radius)
            radius = 3 / np.tan(np.radians(abs(steering_angle)))
            center_x = radius if steering_angle < 0 else -radius
            
            # Generate arc
            angle_range = np.linspace(np.pi/2, 0, len(t)) if steering_angle < 0 else np.linspace(np.pi/2, np.pi, len(t))
            x = center_x + radius * np.cos(angle_range)
            y = radius * np.sin(angle_range)
        
        # Plot trajectory
        ax.plot(x, y, 'b-', linewidth=2, alpha=0.7)
        
        # Load and plot the car image (zaptrack.png)
        car_image = plt.imread('zaptrack.png')
        car_x, car_y = 0, 0.2
        ax.imshow(car_image, extent=(car_x - 0.7, car_x + 0.7, car_y, car_y + 1), aspect='auto', zorder=2)
        # Clean up plot
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])

def main():
    # Set up paths
    model_path = 'best_rc_robot_model.pth'  # Path to your saved model
    image_dir = 'test_images'  # Directory containing test images
    
    # Initialize predictor
    predictor = RCCarPredictor(model_path)
    
    # Process all images in directory
    image_files = Path(image_dir).glob('*.jpg')
    for image_path in image_files:
        print(f"\nProcessing {image_path}...")
        steering, throttle = predictor.predict_image(str(image_path))
        print(f"Predictions:")
        print(f"Steering: {steering:.2f} (1500 is center, <1500 is left, >1500 is right)")
        print(f"Throttle: {throttle:.2f} (1500 is neutral, >1500 is forward, <1500 is reverse)")

if __name__ == "__main__":
    main()