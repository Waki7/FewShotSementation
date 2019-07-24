from Model.Encoders import *
from Model.Decoders import *
import Model.Config as cfg
from os.path import join, isfile
import DataProcessing.DataProcessor as data
import matplotlib.pyplot as plt
import numpy as np
import sys


class SegmentationModel(nn.Module):
    def __init__(self, in_shape, n_class, dilation = 2, size = 1):
        super(SegmentationModel, self).__init__()
        self.encoder = SegEncoder(in_shape=in_shape, out_shape=size, dilation = dilation, size=size)
        self.decoder = SegDecoder(n_class=n_class, n_encoded_channels=self.encoder.out_shape, size = size)

    def forward(self, input):
        encoded_features = self.encoder(input)
        pred = self.decoder(encoded_features)
        return pred

class Segmenter():
    def __init__(self, model: SegmentationModel = None, downsample_ratio=2,
                 lr = .01, size_scale=2, data : data.ProcessedDataSet = None):
        self.size_scale = size_scale
        self.downsample_ratio=downsample_ratio
        self.lr = lr
        self.model_directory = cfg.experiment_path
        self.model_name = 'FullBSRSegmenter'+str(lr)+'.pkl'
        self.model_path = join(self.model_directory, self.model_name)
        self.model = model
        self.data = data
        if model is None:
            self.build_model()

    def build_model(self):
        self.load_data()
        self.classes = int(len(self.class_weights))
        self.model = SegmentationModel(in_shape=self.data.x_shape,
                                       n_class=self.classes,
                                       size=self.size_scale).to(**cfg.args)
        self.opt = torch.optim.SGD(
        self.model.parameters(),
        lr=self.lr,
        momentum=0.9,
        # weight_decay=.001
        )
        self.criterion = nn.NLLLoss(ignore_index=-1, reduction='mean')

    def load_data(self):
        self.data.load_data()
        self.class_weights = torch.tensor(self.data.get_class_weights()).to(**cfg.args)


    def train(self, epochs=10, batch_size=10):
        x_train, y_train = self.data.get_train_data()
        n_train = x_train.shape[0]
        self.model.train()
        mean_loss = 0
        for e in range(epochs):
            shuffled_indexes = torch.randperm(n_train)
            for i in range(0, n_train, batch_size):
                indexes = shuffled_indexes[i:i + batch_size]
                x_batch, y_batch = x_train[indexes], y_train[indexes]
                x_batch, y_batch = torch.tensor(x_batch).to(**cfg.args), torch.tensor(y_batch).to(cfg.device).long()
                y_out = self.model.forward(x_batch)
                loss = self.criterion.forward(input=y_out, target=y_batch)
                mean_loss += loss.data.item()
                loss.backward(retain_graph=False)
                self.opt.step()
                self.opt.zero_grad()
            print(' average loss for epoch ', e, ': ', mean_loss, **cfg.prnt)
            print('val accuracy ', self.pixel_accuracy(440, 460, batch_size), **cfg.prnt)
            print('train accuracy ', self.pixel_accuracy(0, 100, batch_size), **cfg.prnt)
            mean_loss = 0
        return self.model

    def pixel_accuracy(self, start_idx, end_idx, batch_size):
        predictions, ground_truths = self.predict(start_idx, end_idx, batch_size)
        accuracies = []
        for pred, truth in zip(predictions, ground_truths):
            pixel_accuracy = np.average(pred == truth)
            accuracies.append(pixel_accuracy)
        return np.average(accuracies)

    def predict(self, start_idx, end_idx, batch_size):
        self.model.eval()
        predictions = []
        ground_truths = []
        for i in range(start_idx, end_idx, batch_size):
            x_batch = torch.tensor(self.x[i:i + batch_size]).to(**cfg.args)
            prediction = self.model.forward(x_batch)
            prediction = prediction.detach().cpu().numpy()
            prediction = np.argmax(prediction, axis=1)
            prediction = prediction
            ground_truth = self.y[i:i + batch_size]
            predictions.append(prediction)
            ground_truths.append(ground_truth)
        predictions = np.concatenate(predictions, axis=0)
        ground_truths = np.concatenate(ground_truths, axis=0)
        self.model.train()
        return predictions, ground_truths

    def test(self, batch_size):
        print('test accuracy', self.pixel_accuracy(440, 500, batch_size), **cfg.prnt)
        print('accuracy for whole dataset', self.pixel_accuracy(0, 500, batch_size), **cfg.prnt)

    def show_predictions(self, idx):
        self.model.eval()
        _, ax = plt.subplots(1, 2)
        x_batch = torch.tensor(self.x[idx:idx+1]).to(**cfg.args)
        prediction = self.model.forward(x_batch)
        prediction = prediction.detach().cpu().numpy()
        prediction = np.argmax(prediction, axis=1)
        prediction = prediction[0]
        ground_truth = self.y[idx:idx+1][0]
        ax[0].imshow(prediction)
        ax[1].imshow(ground_truth)
        plt.show()
        self.model.train()


    def save_model(self):
        torch.save(self.model, self.model_path)

    def load_model(self):
        if isfile(self.model_path):
            self.model = torch.load(self.model_path)
            return True
        return False



def main():
    print(cfg.lr, **cfg.prnt)
    downsample_ratio = 4
    dataset = data.DataBSR(x_dtype=np.float32, y_dtype=np.int32,
                                                      downsample_ratio=downsample_ratio)
    segmenter = Segmenter(lr=cfg.lr, downsample_ratio=downsample_ratio, size_scale = cfg.model_size, data = dataset)
    if cfg.load:
        segmenter.load_model()
    else:
        segmenter.train(epochs=cfg.epochs, batch_size=cfg.batch_size)
        segmenter.save_model()

    print(segmenter.pixel_accuracy(0, 440, cfg.batch_size))
    print(segmenter.pixel_accuracy(440, 500, cfg.batch_size))
    # segmenter.show_predictions(1)
    print('_______________', **cfg.prnt)


if __name__ == '__main__':
    main()
