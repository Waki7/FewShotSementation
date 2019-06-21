import torch
import torch.nn as nn
import torch.nn.functional as F
import DataProcessing.DataProcessor as data
import numpy as np




class Segmenter(nn.Module):
    def __init__(self, inputShape, classes):
        super(Segmenter, self).__init__()

        numHU = 64
        bias = False
        if len(inputShape) <=2:
            channels = 1
        channels = inputShape[-1] if len(inputShape) > 3 else 1
        strideC_1 = 2
        kernel_size_1 = 3
        strideP_1 = 2
        out_channels_1 = 18

        self.conv1 = torch.nn.Conv2d(in_channels=channels, out_channels = out_channels_1, kernel_size=kernel_size_1,
                                     stride=strideC_1, padding=kernel_size_1)

        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=strideP_1, padding=0)
        self.fc1 = torch.nn.Linear(in_features=inputShape[0]//(strideC_1*strideP_1), out_features=numHU)
        self.fc2 = torch.nn.Linear(numHU, classes)

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
    x, y = torch.tensor(x).cuda(), torch.tensor(y).cuda()
    n_train = x.shape[0]
    classes = np.prod(y.shape[1:])
    model = Segmenter(x.shape, classes)
    opt = torch.optim.SGD(model.parameters(), lr=.1)
    criterion = torch.nn.CrossEntropyLoss()
    shuffled_indexes = torch.randperm(n_train)

    for e in range(epochs):
        for i in range(0, n_train, batch_size):
            indexes = shuffled_indexes[i:i+batch_size]
            x_batch = x[indexes]
            y_batch = y[indexes]
            y_out = model.forward(x_batch)
            loss = criterion(input=y_out, target=y_batch)
            print(loss.data)
            loss.backward(retain_graph=False)
            opt.step()
            opt.zero_grad()

def main():
    ValidateSegmenter()


if __name__ == '__main__':
    main()