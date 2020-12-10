import matplotlib.pyplot as plt

def plot_MNIST_images(imgs, reconstructions):
    nims, nx, ny = imgs.shape
    for i in range(nims):
        plt.subplot(nims, 2, i*2 + 1)
        plt.imshow(imgs[i,:,:])
        plt.subplot(nims, 2, i*2 + 2)
        plt.imshow(reconstructions[i,:,:])

