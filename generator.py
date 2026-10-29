"""
generator.py

Synthetic dataset generator for FRQI-based Quantum Neural Network experiments.
Supports dynamic gradient backgrounds and contextual color-inverting polyps via CLI flag.
"""

import argparse
from pathlib import Path
import random

import cv2
import numpy as np


class SyntheticDatasetGenerator:

    def __init__(
        self,
        image_size=8,
        train_samples=40,
        test_samples=10,
        output_dir="dataset",
        random_seed=22,
        dynamic=False,
    ):
        self.image_size = image_size
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.output_dir = Path(output_dir)
        self.dynamic = dynamic

        random.seed(random_seed)
        np.random.seed(random_seed)

    def generate_background(self):
        """
        Generate a background canvas.
        If dynamic is enabled, generates a smooth gradient matching adjacent pixels.
        Otherwise, uses the classic independent adjacent random values approach.
        """
        if not self.dynamic:
            # Classic implementation
            img = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
            for r in range(self.image_size):
                for c in range(self.image_size):
                    while True:
                        value = random.randint(0, 180)
                        valid = True
                        if r > 0 and img[r - 1, c] == value:
                            valid = False
                        if c > 0 and img[r, c - 1] == value:    
                            valid = False
                        if valid:
                            img[r, c] = value
                            break
            return img

        # Dynamic Gradient Logic: Creates smooth spatial transitions
        # Generate random boundaries for linear interpolation corners
        corners = np.random.randint(0, 181, size=(2, 2)).astype(np.float32)
        
        # Resize corner anchor matrix up to target resolution using bilinear filtering
        img_grad = cv2.resize(corners, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        
        # Inject small local micro-variations to prevent exact floating uniformity
        noise = np.random.uniform(-10, 10, size=img_grad.shape)
        img_smooth = np.clip(img_grad + noise, 0, 180).astype(np.uint8)
        
        return img_smooth

    # -------------------------------------------------------------

    def generate_polyp(self, image):
        """
        Draw a random ellipse representing a polyp.
        If dynamic flag is passed, contextual brightness dictates dark/light inversion rules.
        """
        img = image.copy()
        
        # Geometry coordinates setup
        centre = (
            random.randint(1, self.image_size - 2),
            random.randint(1, self.image_size - 2),
        )
        axis_x = random.randint(1, max(1, self.image_size // 4))
        axis_y = random.randint(1, max(1, self.image_size // 4))
        angle = random.randint(0, 180)

        if not self.dynamic:
            # Classic completely white/bright implementation
            colour = random.randint(181, 255)
            cv2.ellipse(img, centre, (axis_x, axis_y), angle, 0, 360, int(colour), -1)
            return img

        # Contextual Color Inversion Logic
        # Create an isolated temporary canvas mask layer of the proposed ellipse structure
        mask = np.zeros_like(image)
        cv2.ellipse(mask, centre, (axis_x, axis_y), angle, 0, 360, 255, -1)
        
        # Compute mean brightness profile of background covered beneath the mask region
        underlying_pixels = image[mask == 255]
        mean_brightness = np.mean(underlying_pixels) if underlying_pixels.size > 0 else 127.5
        
        # Inversion decision boundary threshold mapping rule
        if mean_brightness >= 90.0:  # Midpoint relative to background limit range [0, 180]
            colour = 0       # Dark element profile overlay for bright regions
        else:
            colour = 255     # Pure bright element profile overlay for dark regions

        # Overwrite selected pixels with matching categorical color property
        img[mask == 255] = colour
        return img


    def validate_image(self, image):
        """
        Ensure adjacent pixels never share the same grayscale value,
        EXCEPT if they are part of a solid polyp region or using dynamic gradients.
        """
        # If dynamic flag is on, structural gradient adjacency sequences are acceptable by design
        if self.dynamic:
            return True

        rows, cols = image.shape
        for r in range(rows):
            for c in range(cols):
                current = image[r, c]
                if r + 1 < rows:
                    neighbor = image[r + 1, c]
                    if current == neighbor and (current <= 180 or neighbor <= 180):
                        return False
                if c + 1 < cols:
                    neighbor = image[r, c + 1]
                    if current == neighbor and (current <= 180 or neighbor <= 180):
                        return False
        return True


    def generate_sample(self, has_polyp):
        while True:
            img = self.generate_background()
            if has_polyp:
                img = self.generate_polyp(img)
            if self.validate_image(img):
                break

        label = 1 if has_polyp else 0
        return img, label


    def save_image(self, image, folder, filename):
        folder.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(folder / filename), image)


    def generate_split(self, split_name, samples):
        positive = samples // 2
        negative = samples // 2
        
        positive_folder = self.output_dir / split_name / "1"
        negative_folder = self.output_dir / split_name / "0"

        print(f"Generating {split_name} split dataset...")

        for i in range(positive):
            img, _ = self.generate_sample(True)
            self.save_image(img, positive_folder, f"polyp_{i:04d}.png")

        for i in range(negative):
            img, _ = self.generate_sample(False)
            self.save_image(img, negative_folder, f"healthy_{i:04d}.png")

    def generate(self):
        print(f"Starting Data Generation Engine (Dynamic Mode = {self.dynamic})")
        self.generate_split("train", self.train_samples)
        self.generate_split("test", self.test_samples)
        print("Dataset generation complete.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic Quantum Image Dataset Pipeline Generator")
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Enable adjacent-aware gradient backgrounds and inverted color polyps"
    )
    parser.add_argument(
        "--size",
        type=int,
        default=8,
        help="Resolution width/height for generation grid context"
    )
    args = parser.parse_args()

    generator = SyntheticDatasetGenerator(
        image_size=args.size,
        dynamic=args.dynamic
    )
    generator.generate()