# Latent Space Explorer

This repository includes a simple variational autoencoder trained on the MNIST dataset.
Each input (a handrawn image from the MNIST dataset) is encoded into a latent space.
Each dimension in the latent space is given by a normally distributed variable.

If training succeeds the latent space is a dense representation of the ten classes of MNIST digits that can be continuously traversed.

To gain an intuition for the learned latent representation, it can be explored with a simple GUI written in tkinter.
Each slider sets the latent location in the latent space along one axis, the image shown is the decoded image represented at the location in the latent space.

(work in progress)

![interface](example.png)
