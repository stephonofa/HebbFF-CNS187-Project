import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch.optim as optim
from torch import save


def visualize_predictions(model, training_data, epoch):
    #  TSNE projects high-dimensional vectors to 2D space
    tsne = TSNE(n_components=2, learning_rate="auto", init="random")

    sample_size = 1000
    data_sampler = DataLoader(training_data, batch_size=sample_size, shuffle=True)
    sample = next(iter(data_sampler))
    images, labels = sample
    feature_vectors = model(images)
    features_2D = tsne.fit_transform(feature_vectors.detach().numpy())

    for digit in range(0, 10):
        points_x, points_y = [], []
        for i in range(len(features_2D)):
            if labels[i].item() == digit:
                points_x.append(features_2D[i, 0])
                points_y.append(features_2D[i, 1])
        plt.scatter(points_x, points_y, label=f"{digit}")

    plt.legend()
    plt.savefig(f"mnist_visualizations/epoch_{epoch}_visualization.png")
    plt.clf()


def model_train_then_val(model, n_epochs, training_data_loader, optimizer, criterion, test_data_loader, train_data):
  # Train the model for n_epochs, iterating on the data in batches

    # store metrics
    training_accuracy_history = np.zeros([n_epochs, 1])
    training_loss_history = np.zeros([n_epochs, 1])
    validation_accuracy_history = np.zeros([n_epochs, 1])
    validation_loss_history = np.zeros([n_epochs, 1])

    for epoch in range(n_epochs):
        visualize_predictions(model, train_data, epoch)
        print(f'Epoch {epoch+1}/{n_epochs}:', end='')
        train_total = 0
        train_correct = 0
        # train
        model.train()
        for i, data in enumerate(training_data_loader):
            images, labels = data
            optimizer.zero_grad()
            # forward pass
            output = model(images)
            # calculate categorical cross entropy loss
            loss = criterion(output, labels)
            # backward pass
            loss.backward()
            optimizer.step()
            
            # track training accuracy
            _, predicted = torch.max(output.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            # track training loss
            training_loss_history[epoch] += loss.item()
            # progress update after 180 batches (~1/10 epoch for batch size 32)
            if i % 180 == 0: print('.',end='')
        training_loss_history[epoch] /= len(training_data_loader)
        training_accuracy_history[epoch] = train_correct / train_total
        print(f'\n\tloss: {training_loss_history[epoch,0]:0.4f}, acc: {training_accuracy_history[epoch,0]:0.4f}',end='')
            
        # validate
        test_total = 0
        test_correct = 0
        with torch.no_grad():
            model.eval()
            for i, data in enumerate(test_data_loader):
                images, labels = data
                # forward pass
                output = model(images)
                # find accuracy
                _, predicted = torch.max(output.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
                # find loss
                loss = criterion(output, labels)
                validation_loss_history[epoch] += loss.item()
            validation_loss_history[epoch] /= len(test_data_loader)
            validation_accuracy_history[epoch] = test_correct / test_total
        print(f', val loss: {validation_loss_history[epoch,0]:0.4f}, val acc: {validation_accuracy_history[epoch,0]:0.4f}')
    if n_epochs == 1:
      return validation_accuracy_history[epoch,0]


BASE_MODEL = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=(3,3), padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),

        nn.Conv2d(32, 32, kernel_size=(3,3), padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        
        nn.Conv2d(32, 32, kernel_size=(3,3), padding=1, stride=2),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Dropout(0.2),

        nn.Conv2d(32, 64, kernel_size=(3,3), padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),

        nn.Conv2d(64, 64, kernel_size=(3,3), padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        
        nn.Conv2d(64, 64, kernel_size=(3,3), padding=1, stride=2),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Dropout(0.2),
        
        nn.Conv2d(64, 128, kernel_size=(3,3), padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Dropout(0.2),
        
        nn.Flatten(),
        nn.Linear(128, 10)
        # PyTorch implementation of cross-entropy loss includes softmax layer
    )


class Encoder(nn.Module):
    def __init__(self, encoding_dim):
        super(Encoder, self).__init__()
        in_features = BASE_MODEL[-1].in_features
        self.encoder = nn.Sequential(
            *list(BASE_MODEL.children())[:-1],
            nn.Linear(in_features, encoding_dim)
        )

        count = len(list(self.encoder.children()))
        for child in self.encoder.children():
            count -= 1
            if count > 0:
                for param in child.parameters():
                    param.requires_grad = False

        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, in_features),
            nn.ReLU(True),
            nn.Linear(in_features, 28*28),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    

if __name__ == "__main__":
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(BASE_MODEL.parameters(), weight_decay=0.01, lr=0.0001)

    train_dataset = datasets.MNIST('./mnist_data', train=True, download=True,  # Downloads into a directory ../data
                                transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
    test_dataset = datasets.MNIST('./mnist_data', train=False, download=False,  # No need to download again
                                transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True) 


    model_train_then_val(BASE_MODEL, 10, train_loader, optimizer, criterion, test_loader, train_dataset)

    encoding_dim = 25
    model = Encoder(encoding_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), weight_decay=0.01, lr=0.001)

    model.train()
    n_epochs = 10
    for epoch in range(n_epochs):
        for images, _ in train_loader:
            
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, images.view(-1, 28 * 28))  # Flatten input images
            loss.backward()
            optimizer.step()
        print('Train Epoch: %d  Loss: %.4f' % (epoch + 1,  loss.item()))

    save(model.state_dict(), "encoder_models/mnist_encoder_state_dict")