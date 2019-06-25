import torch
import torch.nn as nn
import torch.nn.functional as F
import DataProcessing.DataProcessor as data
import matplotlib.pyplot as plt
import numpy as np


use_cpu = False
device = torch.device('cpu') if use_cpu else torch.device('cuda')
type = torch.float32 #if use_cpu else torch.float32 #xentropy doesn't support float16
args = {'device': device, 'dtype': type}


class Segmenter(nn.Module):
    def __init__(self, inputShape, classes=1):
        super(Segmenter, self).__init__()

        numHU = 64
        bias = False
        channels = inputShape[1] if len(inputShape) > 3 else 1
        strideC_1 = 1
        strideP_1 = 1

        kernel_sizeC_1 = 9
        kernel_sizeP_1 = 7
        out_channels_1 = 18

        self.conv1 = torch.nn.Conv2d(in_channels=channels, out_channels=out_channels_1, kernel_size=kernel_sizeC_1,
                                     stride=strideC_1, padding=kernel_sizeC_1 // 2)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=kernel_sizeP_1,
                                        stride=strideP_1, padding=kernel_sizeP_1 // 2)

        strideC_2 = 1
        kernel_sizeC_2 = 9

        self.conv2 = torch.nn.Conv2d(in_channels=out_channels_1, out_channels=classes, kernel_size=kernel_sizeC_2,
                                     # out channels 1 cause labels
                                     stride=strideC_2, padding=kernel_sizeC_2 // 2)

    def forward(self, x):
        # Computes the activation of the first convolution, size will be size of input + padding - kernel size//2
        # get size of memory allocation in bytes : tensorname.element_size() * tensorname.nelement()
        c1 = torch.sigmoid(self.conv1(x))
        p1 = self.pool1(c1)
        c2 = self.conv2(
            p1)
        out = F.softmax(c2, dim=1)  # probabilistic interpretation for last layer classification, we could add channels if multiple labels for each pixel
        # class0 = out.data[0,:,0,0]
        # print(class0.shape)
        # print(torch.sum(class0))
        return out

    def getData(self):
        pass


def ValidateSegmenter():
    epochs = 100
    batch_size = 4

    x, y = data.processBSR(x_dtype = np.float16, y_dtype = np.float16)

    assert not np.any(np.isnan(x))
    assert not np.any(np.isnan(y))
    x = np.transpose(x, (0, 3, 1, 2))
    classes = int(np.max(y)+1)
    print(classes)
    unique, counts = np.unique(y, return_counts=True)
    print(counts)
    print(unique)
    print(sum(counts))
    print(481*481*500)
    print(exit(9))
    n_train = 5#x.shape[0]
    model = Segmenter(x.shape, classes).to(**args)
    opt = torch.optim.Adam(model.parameters(), lr=.00005)
    criterion = torch.nn.CrossEntropyLoss()
    shuffled_indexes = torch.randperm(n_train)
    x, y = torch.tensor(x).to(**args), torch.tensor(y).to(device).long()

    for e in range(epochs):
        for i in range(0, n_train, batch_size):
            indexes = shuffled_indexes[i:i + batch_size]
            x_batch, y_batch = x[indexes], y[indexes]
            # x_batch, y_batch = torch.tensor(x_batch).to(**args), torch.tensor(y_batch).to(device).long()
            y_out = model.forward(x_batch)
            loss = criterion.forward(input=y_out, target=y_batch)
            print(loss.data)
            loss.backward(retain_graph=False)
            opt.step()
            opt.zero_grad()
    #arbitrarily look at the first 10 images and see their output
    for i in range(0, 10):
        _, ax = plt.subplots(1, 2)
        prediction = model.forward(x[i:i+1])
        prediction = prediction.detach().cpu()
        prediction = np.argmax(prediction, axis=1)
        prediction = prediction[0]
        ax[0].imshow(prediction)
        ax[1].imshow(y[i:i+1].cpu()[0])
        plt.show()

def main():
    ValidateSegmenter()


if __name__ == '__main__':
    main()
