from Model.Encoders import *
from Model.Decoders import *
import torch
from torchvision import transforms
import DataProcessing.DataProcessor as data
import matplotlib.pyplot as plt
import numpy as np


class SegmentationModel(nn.Module):
    def __init__(self, in_shape, n_class, dilation = 4):
        super(SegmentationModel, self).__init__()
        self.encoder = SegEncoder(in_shape=in_shape, dilation = 4)
        self.decoder = SegDecoder(n_class=n_class, n_encoded_channels=self.encoder.out_shape)

    def paramters(self):
        return list(self.encoder.parameters()) + list(self.deocder.paramters())

    def forward(self, input):
        encoded_features = self.encoder(input)
        pred = self.decoder(encoded_features)
        return pred

class Segmenter():
    def __init__(self, model: SegmentationModel = None, downsample_ratio=4):
        self.model = model
        if model is None:
            self.build_model(downsample_ratio)

    def build_model(self, downsample_ratio):
        self.load_data(downsample_ratio)
        self.classes = int(len(self.weights))
        self.model = SegmentationModel(in_shape=self.x.shape, n_class=self.classes).to(**args)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=.01)
        self.criterion = torch.nn.CrossEntropyLoss(weight=self.weights)

    def load_data(self, downsample_ratio):
        self.x, x_full, self.y, y_full = data.process_BSR(x_dtype=np.float16, y_dtype=np.int16,
                                                          downsample_ratio=downsample_ratio)
        self.weights = torch.tensor(data.getClassWeights(y_full)).to(**args)


    def train(self, epochs=100, batch_size=2):
        n_train = self.x.shape[0]
        shuffled_indexes = torch.randperm(n_train)
        x, y = torch.tensor(self.x).to(**args), torch.tensor(self.y).to(device).long()
        self.model.train()
        for e in range(epochs):
            for i in range(0, n_train, batch_size):
                indexes = shuffled_indexes[i:i + batch_size]
                x_batch, y_batch = x[indexes], y[indexes]
                # x_batch, y_batch = torch.tensor(x_batch).to(**args), torch.tensor(y_batch).to(device).long()
                y_out = self.model.forward(x_batch)
                loss = self.criterion.forward(input=y_out, target=y_batch)
                print(e,' ', loss.data)
                loss.backward(retain_graph=False)
                self.opt.step()
                self.opt.zero_grad()
        return self.model

    def test(self, seg_model):
        # arbitrarily look at the first 10 images and see their output
        seg_model.eval()
        for i in range(0, 10):
            _, ax = plt.subplots(1, 2)
            prediction = seg_model.forward(x[i:i + 1])
            prediction = prediction.detach().cpu()
            prediction = np.argmax(prediction, axis=1)
            prediction = prediction[0]
            ax[0].imshow(prediction)
            ax[1].imshow(self.y[i:i + 1].cpu()[0])
            plt.show()



    def validate_segmenter(self):
        self.train()
        self.save_model()


def main():
    segmenter = Segmenter()
    segmenter.validate_segmenter()


if __name__ == '__main__':
    main()
