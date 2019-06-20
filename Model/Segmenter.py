import torch
import torch.nn as nn
import torch.nn.functional as F
import DataProcessing.DataProcessor as data




class Segmenter(nn.Module):
    def __init__(self, inputChannels):
        numHU = 8
        bias = False

        # Input channels = 3, output channels = 18
        self.conv1 = torch.nn.Conv2d(3, 18, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # 4608 input features, 64 output features (see sizing flow below)
        self.fc1 = torch.nn.Linear(18 * 16 * 16, 64)

        # 64 input features, 10 output features for our 10 defined classes
        self.fc2 = torch.nn.Linear(64, 10)

    def forward(self, x):
        # Computes the activation of the first convolution
        # Size changes from (3, 32, 32) to (18, 32, 32)
        x = F.relu(self.conv1(x))

        # Size changes from (18, 32, 32) to (18, 16, 16)
        x = self.pool(x)

        # Reshape data to input to the input layer of the neural net
        # Size changes from (18, 16, 16) to (1, 4608)
        # Recall that the -1 infers this dimension from the other given dimension
        x = x.view(-1, 18 * 16 * 16)

        # Computes the activation of the first fully connected layer
        # Size changes from (1, 4608) to (1, 64)
        x = F.relu(self.fc1(x))

        # Computes the second fully connected layer (activation applied later)
        # Size changes from (1, 64) to (1, 10)
        x = self.fc2(x)
        return (x)

    def getData(self):
        pass

def ValidateSegmenter():
    epochs = 10
    batch_size = 100


    x, y = data.processBSR()
    n_train = x.shape[0]
    model = Segmenter(x.shape[-1] if len(x.shape) > 3 else 1)
    opt = torch.optim.SGD(model.parameters(), lr=.1)
    shuffled_indexes = torch.randperm(n_train)

    for e in range(epochs):
        for i in range(0, n_train, batch_size):
            indexes = shuffled_indexes[i:i+batch_size]
            x_batch = x[indexes]
            y_batch = y[indexes]


def main():
    ValidateSegmenter()


if __name__ == '__main__':
    main()