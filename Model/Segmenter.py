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
        strideC_1 = 1
        strideP_1 = 1

        kernel_sizeC_1 = 5
        kernel_sizeP_1 = 3
        out_channels_1 = 18

        self.conv1 = torch.nn.Conv2d(in_channels=channels, out_channels = out_channels_1, kernel_size=kernel_sizeC_1,
                                     stride=strideC_1, padding=kernel_sizeC_1//2)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=kernel_sizeP_1,
                                        stride=strideP_1, padding=kernel_sizeP_1//2)

        strideC_2 = 1
        kernel_sizeC_2 = 7

        self.conv2 = torch.nn.Conv2d(in_channels=out_channels_1, out_channels = 1, kernel_size=kernel_sizeC_2, #out channels 1 cause labels
                                     stride=strideC_2, padding=kernel_sizeC_2//2)

    def forward(self, x):
        # Computes the activation of the first convolution, size will be size of input + padding - kernel size//2
        c1 = F.relu(self.conv1(x))
        p1 = self.pool1(c1)
        c2 = F.sigmoid(self.conv2(p1)) # probabilistic interpretation for last layer classification, we could add channels if multiple labels for each pixel
        return c2

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