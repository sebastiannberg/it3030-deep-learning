import matplotlib.pyplot as plt
import torch


def visualize_images(images: torch.Tensor, targets: torch.Tensor):
    num_images = images.size(0)
    grid_size = int(num_images**0.5)
    fig, ax = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    for i in range(num_images):
        ax[i // grid_size, i % grid_size].imshow(images[i].squeeze(), cmap='gray')
        ax[i // grid_size, i % grid_size].title.set_text(str(targets[i].item()))
        ax[i // grid_size, i % grid_size].axis('off')
    plt.show()

def visualize_reconstructions(original_images: torch.Tensor, reconstructions: torch.Tensor, targets: torch.Tensor, max_images=4):
    num_images = min(original_images.size(0), max_images)

    fig, axes = plt.subplots(nrows=num_images, ncols=2, figsize=(5, 2*num_images))

    # If there's only one image, axes array is not subscriptable
    if num_images == 1:
        axes = [axes]

    for i in range(num_images):
        # Original Image
        ax = axes[i][0]
        ax.imshow(original_images[i].squeeze(), cmap='gray')
        ax.title.set_text('Original - ' + str(targets[i].item()))
        ax.axis('off')

        # Reconstructed Image
        ax = axes[i][1]
        ax.imshow(reconstructions[i].squeeze(), cmap='gray')
        ax.title.set_text('Reconstructed - ' + str(targets[i].item()))
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def visualize_generated_examples(generated_examples):
    pass
