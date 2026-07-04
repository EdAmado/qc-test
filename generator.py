"""
generator.py

Synthetic dataset generator for FRQI-based Quantum Neural Network experiments.
"""

from pathlib import Path
import random

import cv2
import numpy as np


class SyntheticDatasetGenerator:

    def __init__(
        self,
        image_size=4,
        train_samples=40,
        test_samples=10,
        output_dir="dataset",
        random_seed=42,
    ):

        self.image_size = image_size
        self.train_samples = train_samples
        self.test_samples = test_samples

        self.output_dir = Path(output_dir)

        random.seed(random_seed)
        np.random.seed(random_seed)

    # -------------------------------------------------------------

    def generate_background(self):
        """
        Generate a random grayscale background where
        no adjacent pixels have the same value.
        """

        img = np.zeros(
            (self.image_size, self.image_size),
            dtype=np.uint8,
        )

        for r in range(self.image_size):
            for c in range(self.image_size):

                while True:

                    value = random.randint(0, 180)

                    valid = True

                    if r > 0:
                        if img[r - 1, c] == value:
                            valid = False

                    if c > 0:
                        if img[r, c - 1] == value:
                            valid = False

                    if valid:
                        img[r, c] = value
                        break

        return img

    # -------------------------------------------------------------

    def generate_polyp(self, image):
        """
        Draw a random ellipse representing a polyp.

        Returns
        -------
        image
        """

        img = image.copy()

        colour = random.randint(181, 255)

        centre = (
            random.randint(1, self.image_size - 2),
            random.randint(1, self.image_size - 2),
        )

        axis_x = random.randint(1, max(1, self.image_size // 4))
        axis_y = random.randint(1, max(1, self.image_size // 4))

        angle = random.randint(0, 180)

        cv2.ellipse(
            img,
            centre,
            (axis_x, axis_y),
            angle,
            0,
            360,
            int(colour),
            -1,
        )

        return img

    # -------------------------------------------------------------

    def validate_image(self, image):
        """
        Ensure adjacent pixels never share the same grayscale value,
        EXCEPT if they are part of the solid polyp (values > 180).
        """
        rows, cols = image.shape

        for r in range(rows):
            for c in range(cols):
                current = image[r, c]

                # Check pixel below
                if r + 1 < rows:
                    neighbor = image[r + 1, c]
                    # Only fail if they match AND at least one of them is background
                    if current == neighbor and (current <= 180 or neighbor <= 180):
                        return False

                # Check pixel to the right
                if c + 1 < cols:
                    neighbor = image[r, c + 1]
                    # Only fail if they match AND at least one of them is background
                    if current == neighbor and (current <= 180 or neighbor <= 180):
                        return False

        return True

    # -------------------------------------------------------------

    def generate_sample(self, has_polyp):
        """
        Generate one image and label.
        """

        while True:

            img = self.generate_background()

            if has_polyp:
                img = self.generate_polyp(img)

            if self.validate_image(img):
                break

        label = 1 if has_polyp else 0

        return img, label

    # -------------------------------------------------------------

    def save_image(self, image, folder, filename):

        folder.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(
            str(folder / filename),
            image,
        )

    # -------------------------------------------------------------

    def generate_split(self, split_name, samples):

        positive = samples // 2
        negative = samples // 2

        positive_folder = (
            self.output_dir
            / split_name
            / "1"
        )

        negative_folder = (
            self.output_dir
            / split_name
            / "0"
        )

        print(f"\nGenerating {split_name} dataset...")

        for i in range(positive):

            img, _ = self.generate_sample(True)

            self.save_image(
                img,
                positive_folder,
                f"polyp_{i:04d}.png",
            )

        for i in range(negative):

            img, _ = self.generate_sample(False)

            self.save_image(
                img,
                negative_folder,
                f"healthy_{i:04d}.png",
            )

    # -------------------------------------------------------------

    def generate(self):

        self.generate_split(
            "train",
            self.train_samples,
        )

        self.generate_split(
            "test",
            self.test_samples,
        )

        print("\nDataset generation complete.")


# ----------------------------------------------------------------

if __name__ == "__main__":

    generator = SyntheticDatasetGenerator(
        image_size=4,
        train_samples=500,
        test_samples=100,
    )

    generator.generate()