import scipy.io
import os
from os.path import join, isfile
import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt
from scipy.io import loadmat


class DataSet():
    def __init__(self):
        self.fileExt = None
        self.rootPath = None
        self.processedDataPath = None
        self.dataFileName = None
        self.downDataFileName = None
        self.paths = None
        self.downsampleRatio = 1

    def readFile(self, file):
        raise NotImplementedError

    def pad(self, image: np.ndarray, maxShape):
        pad_width = [(0, maxShape[i] - image.shape[i]) for i in range(len(maxShape))]
        return np.pad(image,
                      pad_width=pad_width, mode='constant')

    def saveData(self, path, filename, array):
        if not os.path.exists(path):
            os.makedirs(path)
        with open(path + filename, 'wb') as f: pickle.dump(array, f)

    def loadArray(self, path, filename):
        with open(path + filename, 'rb') as f: return pickle.load(f)

    def processImages(self, directoryPath, pad = True):
        data = []
        data_downsampled = []
        maxShape = None
        for filename in os.listdir(directoryPath):
            if filename.endswith(self.fileExt):
                samplePath = join(directoryPath, filename)
                image, image_downsampled = self.readFile(samplePath)
                if maxShape is None:
                    maxShape = [0]*len(image.shape)
                for i in range(len(image.shape)):
                    maxShape[i] = max(image.shape[i], maxShape[i])
                data.append(image)
                data_downsampled.append(image_downsampled)
        if pad:
            data = [self.pad(im, maxShape = maxShape) for im in data]
            data_downsampled = [self.pad(im, maxShape=[i//self.downsampleRatio for i in maxShape]) for im in data_downsampled]
        return np.stack(data), np.stack(data_downsampled)

    def getData(self):
        data = []
        data_downsampled = []
        if isfile(self.processedDataPath + self.dataFileName) and isfile(self.processedDataPath + self.downDataFileName):
            data, data_downsampled = self.loadArray(self.processedDataPath, self.dataFileName)
            print('...loaded array of shape ' + str(data.shape))
            return data, data_downsampled
        for set in self.paths:
            processed = self.processImages(set, pad=True)
            data.append(processed[0])
            data_downsampled.append(processed[1])
        data = np.vstack(data)
        data_downsampled = np.vstack(data_downsampled)
        self.saveData(self.processedDataPath, self.dataFileName, data)
        self.saveData(self.processedDataPath, self.downDataFileName, data_downsampled)
        return data, data_downsampled

class BSRLabels(DataSet):
    def __init__(self, downsampleRatio):
        super(BSRLabels, self).__init__()
        self.fileExt = '.mat'
        self.rootPath = '..\\Data\\BSR\\BSDS500\\data\\groundTruth\\'
        self.processedDataPath = self.rootPath + 'ProcessedData\\'
        self.dataFileName = 'processedLabels.pkl'
        self.downDataFileName = 'downsampledLabels.pkl'
        self.paths = [self.rootPath + i for i in ['train', 'test', 'val']]
        self.matKey = 'groundTruth'
        self.segmentationIndex = 0
        self.boundaryIndex = 1
        self.downsampleRatio = downsampleRatio

    def readFile(self, file):
        mat = scipy.io.loadmat(file)
        mat_data = np.squeeze(mat[self.matKey][0, 0]).item(0)
        datum = mat_data[self.segmentationIndex] #segementation ground truth, mat_data[1] is the boundary boxes
        datum_downsampled = downsample(datum, ratio=self.downsampleRatio)
        # datum1 = mat_data[1]
        # plt.imshow(datum)
        # plt.show()
        return datum, datum_downsampled

class BSRImages(DataSet):
    def __init__(self, downsampleRatio):
        super(BSRImages, self).__init__()
        self.fileExt = '.jpg'
        self.rootPath = '..\\Data\\BSR\\BSDS500\\data\\images\\'
        self.processedDataPath = self.rootPath + 'ProcessedData\\'
        self.dataFileName = 'processedImages.pkl'
        self.downDataFileName = 'downsampledImages.pkl'
        self.paths = [self.rootPath + i for i in ['train', 'test', 'val']]
        self.downsampleRatio = downsampleRatio

    def readFile(self, file):
        datum = scipy.misc.imread(file)
        datum_downsampled = downsample(datum, ratio=self.downsampleRatio)
        return datum, datum_downsampled


def processBSR(x_dtype = np.float16, y_dtype = np.float16, downsampleRatio=4): # c x h x w
    labels = BSRLabels(downsampleRatio)
    images = BSRImages(downsampleRatio)
    x = images.getData()
    y = labels.getData()
    if x.dtype != x_dtype:
        x = x.astype(x_dtype)
    if y.dtype != y_dtype:
        y = y.astype(y_dtype)
    x = cleanInput(x)
    assert not np.any(np.isnan(x))
    assert not np.any(np.isnan(y))
    return x, y

def downsample(img, ratio):
    newH = img.shape[0] // ratio
    newW = img.shape[1] // ratio
    img = cv2.resize(img, (newH, newW), interpolation=cv2.INTER_NEAREST)
    return img

def main():
    x, y = processBSR()


if __name__=='__main__':
    main()


def cleanInput(x):
    print('...reshaped from ', x.shape)
    if len(x.shape) > 3:
        x = np.transpose(x, (3, 2, 0, 1))
        x = x / np.max(x)  # scale [0,255] -> [0,1]
    print('to ', x.shape)
    return x


def getClassWeights(y):
    unique, counts = np.unique(y, return_counts=True)
    totalCount = sum(counts)
    return [totalCount/c for c in counts]