from Model.Encoders import *
import torch
from torchvision import transforms
import DataProcessing.DataProcessor as data
import matplotlib.pyplot as plt
import numpy as np




def ValidateSegmenter():
    epochs = 1000
    batch_size = 1

    x, y = data.processBSR(x_dtype = np.float16, y_dtype = np.float16)
    x = data.cleanInput(x)
    weights = torch.tensor(data.getClassWeights(y)).to(**args)
    classes = int(np.max(y)+1)
    n_train = 1 #x.shape[0]
    model = SegEncoder(in_shape = x.shape, n_class=classes).to(**args)
    opt = torch.optim.Adam(model.parameters(), lr=.01)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
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
