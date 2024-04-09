import matplotlib.pyplot as plt
import torch


def visualize_images(title: str, images: torch.Tensor, targets: torch.Tensor):
    num_images = images.size(0)
    grid_size = int(num_images**0.5)
    fig, ax = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    for i in range(num_images):
        ax[i // grid_size, i % grid_size].imshow(images[i].squeeze(), cmap='gray')
        ax[i // grid_size, i % grid_size].title.set_text(str(targets[i].item()))
        ax[i // grid_size, i % grid_size].axis('off')
    fig.suptitle(title)
    plt.show()

def visualize_reconstructions(original_images: torch.Tensor, reconstructions: torch.Tensor, max_images=5):
    for i, (img, reconstruction) in enumerate(zip(original_images, reconstructions)):
        if i >= max_images:
            break
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
        axes[0].imshow(img.squeeze(), cmap='gray')
        axes[0].title.set_text("Original")
        axes[0].axis('off')
        axes[1].imshow(reconstruction.squeeze(), cmap='gray')
        axes[1].title.set_text("Reconstruction")
        axes[1].axis('off')
    plt.show()

def visualize_generated_examples(generated_examples):
    pass
