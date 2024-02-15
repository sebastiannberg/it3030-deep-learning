import matplotlib.pyplot as plt
import numpy as np


class Visualizer:

    def view_images(self, cases, num_images):
        # Assuming cases come in 5-item tuples
        images, _, labels, dims, flat = cases  # Unpack the 5-item tuple
        if flat:
            images = np.array([a.reshape(*dims) for a in images])

        # Choose random images
        random_indices = np.random.choice(len(images), num_images, replace=False)
        random_images = images[random_indices]
        random_labels = [labels[i] for i in random_indices]

        for image, label in zip(random_images, random_labels):
            plt.figure()
            plt.imshow(image, cmap="gray")
            plt.title(f"Label: {label}")

        plt.show()

    def plot_learning_progression(self, train_loss, validation_loss):
        plt.figure()
        plt.title("Learning Progression")
        plt.xlabel("Minibatch")
        plt.ylabel("Loss")
        plt.plot(train_loss, label="Train", color="blue")
        x = np.arange(len(train_loss))
        slope, intercept = np.polyfit(x, train_loss, 1)
        y = slope * x + intercept
        plt.plot(x, y, linestyle='--', color='gray')
        plt.legend()

        plt.figure()
        plt.title("Learning Progression")
        plt.xlabel("Minibatch")
        plt.ylabel("Loss")
        plt.plot(validation_loss, label="Validation", color="orange")
        x = np.arange(len(validation_loss))
        slope, intercept = np.polyfit(x, validation_loss, 1)
        y = slope * x + intercept
        plt.plot(x, y, linestyle='--', color='gray')
        plt.legend()

        plt.show()
