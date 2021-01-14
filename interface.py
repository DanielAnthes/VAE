from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk) 
import matplotlib.pyplot as plt
import tkinter as tk
import numpy as np
import tensorflow as tf
from vae import VAE

class Interface:

    def __init__(self):

        self.n_latent = 16

        self.root = tk.Tk()
        self.root.geometry("800x1200")

        self.model = VAE(n_latent=16)
        self.model.load()

        self.latent_activations = np.zeros((1,16))
        img = self.model.decode(self.latent_activations)

        self.imfig = plt.figure()
        self.imax = self.imfig.add_subplot(111)
        self.imax.set_axis_off()
        self.imax.imshow(img.numpy().squeeze())

        canvas = FigureCanvasTkAgg(self.imfig, master=self.root)
        canvas.draw()
        canvas.get_tk_widget().pack()

        btn = tk.Button(master=self.root, text="randomize", command=self.randomize)
        btn.pack()

        self.sliders = list()
        for i in range(self.n_latent):
            slider = tk.Scale(master = self.root, command=self.update_latent, from_=-2, to=2, orient=tk.HORIZONTAL, resolution=0.01, length=500)
            slider.pack()
            self.sliders.append(slider)

        self.randomize()
        self.root.mainloop()


    def update_image(self):
        img = self.model.decode(self.latent_activations)
        img = img.numpy().squeeze()
        self.imax.imshow(img)
        self.imfig.canvas.draw()

    def randomize(self):
        self.latent_activations = np.random.random((1,16))
        for i in range(self.n_latent):
            self.sliders[i].set(self.latent_activations[0,i])
        self.update_image()

    def update_latent(self, _):
        for i in range(self.n_latent):
            self.latent_activations[0,i] = self.sliders[i].get()
        self.update_image()

inter = Interface()
