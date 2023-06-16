import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torchvision.models import resnet18
import torchvision.transforms as transforms
from torch import save
import torch.optim as optim

BASE_MODEL = resnet18(weights="DEFAULT")
TRANSFORM = transforms.Compose(
    [transforms.ToPILImage(),
     transforms.Resize((224,224)), 
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # ImageNet normalization
     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])] # Normalize to range [-1, 1]
)


class ImageDataset(Dataset):
    def __init__(self, X) -> None:
        self.X = X
        self.transform = TRANSFORM

    def __len__(self) -> None:
        return len(self.X)

    def __getitem__(self, index):
        return self.transform(self.X[index])
    

class Autoencoder(nn.Module):
    def __init__(self, encoding_dim, img_dim, n_frzn_layers):
        super(Autoencoder, self).__init__()
        in_features = BASE_MODEL.fc.in_features
        self.encoder = nn.Sequential(
            *list(BASE_MODEL.children())[:-1],
            nn.Flatten(),
            nn.Linear(in_features, encoding_dim)
        )
        count = 0
        for child in self.encoder.children():
            count += 1
            if count <= n_frzn_layers:
                for param in child.parameters():
                    param.requires_grad = False

        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, in_features),
            nn.ReLU(True),
            nn.Linear(in_features, 3*img_dim*img_dim),
            nn.Tanh()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    

if __name__ == "__main__":
    encoding_dim = 25
    model = Autoencoder(encoding_dim, 224, 9)
    X = np.load("loaded_imgs/loaded_OBJECTSALL.npy")
    X2 = np.load("loaded_imgs/loaded_STATE.npy")
    X3 = np.concatenate((X,X2), axis=0)
    dataset = ImageDataset(X3)
    dataloader = DataLoader(dataset, batch_size=26, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    model.train()
    n_epochs = 10
    for epoch in range(n_epochs):
        for images in dataloader:
            
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, images.view(-1, 3 * 224 * 224))  # Flatten input images
            loss.backward()
            optimizer.step()
        print('Train Epoch: %d  Loss: %.4f' % (epoch + 1,  loss.item()))

    save(model.state_dict(), f"encoder_models/image_encoder{encoding_dim}_state_dict")

