from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk) 
import matplotlib.pyplot as plt
import tkinter as tk
import numpy as np
import tensorflow as tf
from vae import VAE

class Interface:

    def __init__(self):
        self.root = tk.Tk()
        self.root.geometry("1000x1000")

        self.model = VAE(n_latent=16)
        self.model.load()

        latent_activations = np.random.random((1,16))
        img = self.model.decode(latent_activations)

        self.imfig = plt.figure()
        self.imax = self.imfig.add_subplot(111)
        self.imax.set_axis_off()
        self.imax.imshow(img.numpy().squeeze())

        canvas = FigureCanvasTkAgg(self.imfig, master=self.root)
        canvas.draw()
        canvas.get_tk_widget().pack()

        btn = tk.Button(master=self.root, text="randomize", command=lambda: self.update_image(self.model.decode(np.random.random((1,16))).numpy().squeeze()))
        btn.pack()

        self.root.mainloop()


    def update_image(self, data):
        self.imax.imshow(data)
        self.imfig.canvas.draw()

inter = Interface()
