from Model.Encoders import *
from Model.Decoders import *
from Model.Config import *
from os.path import isfile
import DataProcessing.DataProcessor as data
import matplotlib.pyplot as plt
import numpy as np


class SegmentationModel(nn.Module):
    def __init__(self, in_shape, n_class, dilation = 2):
        super(SegmentationModel, self).__init__()
        self.encoder = SegEncoder(in_shape=in_shape, out_shape=256, dilation = dilation)
        self.decoder = SegDecoder(n_class=n_class, n_encoded_channels=self.encoder.out_shape)

    def paramters(self):
        return list(self.encoder.parameters()) + list(self.deocder.paramters())

    def forward(self, input):
        encoded_features = self.encoder(input)
        pred = self.decoder(encoded_features)
        return pred

class Segmenter():
    def __init__(self, model: SegmentationModel = None, downsample_ratio=4):
        self.downsample_ratio=downsample_ratio
        self.model_path = '..\\StoredModels\\'
        self.model_name = 'FullBSRSegmenter.pkl'
        self.model = model
        if model is None:
            self.build_model()

    def build_model(self):
        self.load_data()
        self.classes = int(len(self.weights))
        self.model = SegmentationModel(in_shape=self.x.shape, n_class=self.classes).to(**args)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=.01)
        self.criterion = torch.nn.CrossEntropyLoss(weight=self.weights)

    def load_data(self, ):
        self.x, x_full, self.y, y_full = data.process_BSR(x_dtype=np.float32, y_dtype=np.int32,
                                                          downsample_ratio=self.downsample_ratio)
        self.weights = torch.tensor(data.getClassWeights(y_full)).to(**args)


    def train(self, epochs=10, batch_size=25):
        n_train = 400#self.x.shape[0]
        self.model.train()
        mean_loss = 0
        for e in range(epochs):
            shuffled_indexes = torch.randperm(n_train)
            for i in range(0, n_train, batch_size):
                indexes = shuffled_indexes[i:i + batch_size]
                x_batch, y_batch = self.x[indexes], self.y[indexes]
                x_batch, y_batch = torch.tensor(x_batch).to(**args), torch.tensor(y_batch).to(device).long()
                y_out = self.model.forward(x_batch)
                loss = self.criterion.forward(input=y_out, target=y_batch)
                mean_loss += loss.data.item()
                loss.backward(retain_graph=False)
                self.opt.step()
                self.opt.zero_grad()
            print(' average loss for epoch ', e, ': ', mean_loss / n_train)
            mean_loss = 0
        return self.model

    def test(self):
        # arbitrarily look at the first 10 images and see their output
        self.model.eval()
        averages = []
        for i in range(400, 500):
            _, ax = plt.subplots(1, 2)
            x_batch = torch.tensor(self.x[i:i + 1]).to(**args)
            prediction = self.model.forward(x_batch)
            prediction = prediction.detach().cpu().numpy()
            prediction = np.argmax(prediction, axis=1)
            prediction = prediction[0]
            ground_truth = self.y[i:i + 1][0]
            ax[0].imshow(prediction)
            ax[1].imshow(ground_truth)
            plt.show()
            pixel_accuracy = np.average(prediction == ground_truth)
            print(pixel_accuracy)
            averages.append(pixel_accuracy)
        print(averages)

    def save_model(self):
        torch.save(self.model, self.model_path)

    def load_model(self):
        if isfile(self.model_path):
            self.model = torch.load(self.model_path)
            return True
        return False



def main():
    segmenter = Segmenter()
    segmenter.train(epochs=100)
    segmenter.save_model()
    segmenter.test()


if __name__ == '__main__':
    main()
