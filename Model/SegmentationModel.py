from Model.Encoders import *
from Model.Decoders import *
import Model.Config as cfg
from os.path import join, isfile
import DataProcessing.DataProcessor as data
import matplotlib.pyplot as plt
import numpy as np
import sys


class SegmentationModel(nn.Module):
    def __init__(self, in_shape, n_class, dilation = 2, size_scale = 1):
        super(SegmentationModel, self).__init__()
        base_size = 64
        size = int(base_size * size_scale)
        self.encoder = SegEncoder(in_shape=in_shape, out_shape=size, dilation = dilation, size=size)
        self.decoder = SegDecoder(n_class=n_class, n_encoded_channels=self.encoder.out_shape, size = size)

    def forward(self, input):
        encoded_features = self.encoder(input)
        pred = self.decoder(encoded_features)
        return pred

class Segmenter():
    def __init__(self, model: SegmentationModel = None, downsample_ratio=2, lr = .01, size_scale=2):
        self.size_scale = size_scale
        self.downsample_ratio=downsample_ratio
        self.lr = lr
        self.model_directory = '..\\StoredModels\\'
        self.model_name = 'FullBSRSegmenter'+str(lr)+'.pkl'
        self.model_path = join(self.model_directory, self.model_name)
        self.model = model
        if model is None:
            self.build_model()

    def build_model(self):
        self.load_data()
        self.classes = int(len(self.weights))
        self.model = SegmentationModel(in_shape=self.x.shape,
                                       n_class=self.classes,
                                       size_scale=self.size_scale).to(**cfg.args)
        self.opt = torch.optim.SGD(
        self.model.parameters(),
        lr=self.lr,
        momentum=0.9,
        weight_decay=.0001)
        self.criterion = nn.NLLLoss(ignore_index=-1)

    def load_data(self, ):
        self.x, x_full, self.y, y_full = data.process_BSR(x_dtype=np.float32, y_dtype=np.int32,
                                                          downsample_ratio=self.downsample_ratio)
        self.weights = torch.tensor(data.getClassWeights(y_full)).to(**cfg.args)


    def train(self, epochs=10, batch_size=10):
        n_train = 440#self.x.shape[0]
        self.model.train()
        mean_loss = 0
        for e in range(epochs):
            shuffled_indexes = torch.randperm(n_train)
            for i in range(0, n_train, batch_size):
                indexes = shuffled_indexes[i:i + batch_size]
                x_batch, y_batch = self.x[indexes], self.y[indexes]
                x_batch, y_batch = torch.tensor(x_batch).to(**cfg.args), torch.tensor(y_batch).to(cfg.device).long()
                y_out = self.model.forward(x_batch)
                loss = self.criterion.forward(input=y_out, target=y_batch)
                mean_loss += loss.data.item()
                loss.backward(retain_graph=False)
                self.opt.step()
                self.opt.zero_grad()
            print(' average loss for epoch ', e, ': ', mean_loss / (n_train//batch_size+1), **cfg.prnt)
            print('val accuracy ', self.pixel_accuracy(440, 460), **cfg.prnt)
            print('train accuracy ', self.pixel_accuracy(0, 100), **cfg.prnt)
            mean_loss = 0
        return self.model

    def pixel_accuracy(self, start_idx, end_idx):
        self.model.eval()
        averages = []
        for i in range(start_idx, end_idx):
            x_batch = torch.tensor(self.x[i:i + 1]).to(**cfg.args)
            prediction = self.model.forward(x_batch)
            prediction = prediction.detach().cpu().numpy()
            prediction = np.argmax(prediction, axis=1)
            prediction = prediction[0]
            ground_truth = self.y[i:i + 1][0]
            pixel_accuracy = np.average(prediction == ground_truth)
            averages.append(pixel_accuracy)
        self.model.train()
        return np.average(averages)

    def test(self):
        # arbitrarily look at the first 10 images and see their output
        self.model.eval()
        averages = []
        for i in range(440, 500):
            # _, ax = plt.subplots(1, 2)
            x_batch = torch.tensor(self.x[i:i + 1]).to(**cfg.args)
            prediction = self.model.forward(x_batch)
            prediction = prediction.detach().cpu().numpy()
            prediction = np.argmax(prediction, axis=1)
            prediction = prediction[0]
            ground_truth = self.y[i:i + 1][0]
            # ax[0].imshow(prediction)
            # ax[1].imshow(ground_truth)
            # plt.show()
            pixel_accuracy = np.average(prediction == ground_truth)
            averages.append(pixel_accuracy)
        print(averages, **cfg.prnt)
        print(np.average(averages), **cfg.prnt)

    def save_model(self):
        torch.save(self.model, self.model_path)

    def load_model(self):
        if isfile(self.model_path):
            self.model = torch.load(self.model_path)
            return True
        return False



def main():
    lrs = [.01, .1, 1] # .05 dece, probably bigger model needed?
    for lr in lrs:
        print(lr, **cfg.prnt)
        segmenter = Segmenter(lr=lr, downsample_ratio=4, size_scale = 2)
        segmenter.train(epochs=500, batch_size=40)
        segmenter.save_model()
        segmenter.test()
        print('_______________', **cfg.prnt)


if __name__ == '__main__':
    main()
