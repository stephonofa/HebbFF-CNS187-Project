import torch
from mnist_encoder import Encoder
from image_encoder import Autoencoder, ImageDataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from torch.nn.functional import sigmoid
import pickle

def load_mnist_encodings(enc_dim):
    model = Encoder(enc_dim)
    model.load_state_dict(torch.load("./encoder_models/mnist_encoder_state_dict"))
    model.eval()

    dataset = datasets.MNIST('./mnist_data', train=True, download=True,  # Downloads into a directory ../data
                                transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
    dataloader = DataLoader(dataset, batch_size=1)

    encoded_images = {}
    for i in range(10):
        encoded_images[str(i)] = []

    with torch.no_grad():
        for images, label in dataloader:
            encoded = model.encoder(images)
            encoded_images[str(label.detach().numpy()[0])].append(torch.round(sigmoid(encoded)).detach().numpy()[0])

    with open(f'./loaded_encodings/mnist_d={enc_dim}_encodings.pkl', 'wb') as fp:
        pickle.dump(encoded_images, fp)


def load_image_encodings(enc_dim, img_dim, img_type):
    model = Autoencoder(enc_dim, img_dim, 9)
    model.load_state_dict(torch.load(f"encoder_models/image_encoder{enc_dim}_state_dict"))
    model.eval()

    X = np.load(f"loaded_imgs/loaded_{img_type}.npy")
    dataset = ImageDataset(X)
    dataloader = DataLoader(dataset, batch_size=1)
    encoded_images = []
    with torch.no_grad():
        for images in dataloader:
            encoded = model.encoder(images)
            encoded_images.append(torch.round(sigmoid(encoded)).detach().numpy()[0])

    encoded_images = np.array(encoded_images)
    np.save(f"./loaded_encodings/{img_type}_d={enc_dim}_encodings.npy", encoded_images)

if __name__ == '__main__':
    load_mnist_encodings(25)
    load_image_encodings(25, 224, "STATE")
    load_image_encodings(25, 224, "OBJECTSALL")