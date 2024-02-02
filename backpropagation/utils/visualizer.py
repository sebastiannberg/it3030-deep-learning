import matplotlib.pyplot as plt
import numpy as np


class Visualizer:

    def view_images(self, cases, num_images):
        # Assuming cases come in 5-item tuples
        images, _, labels, dims, flat = cases  # Unpack the 5-item tuple
        if flat:
            images = [a.reshape(*dims) for a in images]

        # Choose random images
        random_indices = np.random.choice(
            len(images), num_images, replace=False)
        random_images = images[random_indices]
        random_labels = [labels[i] for i in random_indices]

        for image, label in zip(random_images, random_labels):
            plt.figure()
            plt.imshow(image, cmap="gray")
            plt.title(f"Label: {label}")

        plt.show()

    def plot_learning_progress(self, train_loss, validation_loss):
        pass
