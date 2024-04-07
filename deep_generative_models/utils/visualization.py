import matplotlib.pyplot as plt


def visualize_dataset(data_loader, num_images):
    grid_size = int(num_images**0.5)
    fig, ax = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    for images, labels in data_loader:
        for i in range(num_images):
            ax[i // grid_size, i % grid_size].imshow(images[i].squeeze(), cmap='gray')
            ax[i // grid_size, i % grid_size].title.set_text(str(labels[i].item()))
            ax[i // grid_size, i % grid_size].axis('off')
        break
    plt.show()