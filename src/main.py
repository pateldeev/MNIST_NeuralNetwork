import data_loader as dl
import network

import cv2
import numpy as np

# train_data = dl.get_images("/home/dp/PycharmProjects/MNIST_Classification_Network/data/train-images-idx3-ubyte")
# dl.save_images(train_data, "/home/dp/PycharmProjects/MNIST_Classification_Network/data/train/")
train_data = dl.read_images("/home/dp/PycharmProjects/MNIST_Classification_Network/data/train/", read_limit=3)

train_labels = dl.get_labels("/home/dp/PycharmProjects/MNIST_Classification_Network/data/train-labels-idx1-ubyte")

net = network.Network()

# net.randomize_parameters()
# net.save_parameters("/home/dp/PycharmProjects/MNIST_Classification_Network/data/")

net.read_parameters("/home/dp/PycharmProjects/MNIST_Classification_Network/data/")

print(net.compute_output(train_data[0]))

for _ in range(10):
    net.back_prop(train_data[0], train_labels[0])
