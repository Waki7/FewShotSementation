import scipy.io
import os
from os.path import join, isfile
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.io import loadmat


class DataSet():
    def __init__(self):
        self.fileExt = None

    def readFile(self, file):
        raise NotImplementedError

    def pad(self, image: np.ndarray, maxShape):
        pad_width = [(0, maxShape[i] - image.shape[i]) for i in range(len(maxShape))]
        return np.pad(image,
                      pad_width=pad_width, mode='constant')

    def saveData(self, path, filename, array, compress=True):
        if not os.path.exists(path):
            os.makedirs(path)
        if compress:
            if np.issubdtype(array.dtype, np.integer):
                array = array.astype(np.int8)
                print(array.dtype)
            if np.issubdtype(array.dtype, np.floating):
                array = array.astype(np.float16)
        with open(path + filename, 'wb') as f: pickle.dump(array, f)

    def loadArray(self, path, filename):
        with open(path + filename, 'rb') as f: return pickle.load(f)

    def processImages(self, directoryPath, pad = True):
        data = []
        maxShape = None
        for filename in os.listdir(directoryPath):
            if filename.endswith(self.fileExt):
                samplePath = join(directoryPath, filename)
                image = self.readFile(samplePath)
                if maxShape is None:
                    maxShape = [0]*len(image.shape)
                for i in range(len(image.shape)):
                    maxShape[i] = max(image.shape[i], maxShape[i])
                data.append(self.readFile(samplePath))
        if pad:
            data = [self.pad(im, maxShape = maxShape) for im in data]
        return np.stack(data)


class BSRLabels(DataSet):
    def __init__(self):
        super(BSRLabels, self).__init__()
        self.fileExt = '.mat'
        self.rootPath = '..\\Data\\BSR\\BSDS500\\data\\groundTruth\\'
        self.processedDataPath = self.rootPath + 'ProcessedData\\'
        self.processedDataName = 'processedLabels.pkl'
        self.paths = [self.rootPath + i for i in ['train', 'test', 'val']]
        self.matKey = 'groundTruth'
        self.segmentationIndex = 0
        self.boundaryIndex = 1

    def readFile(self, file):
        mat = scipy.io.loadmat(file)
        mat_data = np.squeeze(mat[self.matKey][0, 0]).item(0)
        datum = mat_data[self.segmentationIndex] #segementation ground truth, mat_data[1] is the boundary boxes
        return datum

    def getData(self):
        data = []
        if isfile(self.processedDataPath + self.processedDataName):
            data = self.loadArray(self.processedDataPath, self.processedDataName)
            print('...loaded array of shape ' + str(data.shape))
            return data
        for set in self.paths:
            data.append(self.processImages(set, pad=True))
        data = np.vstack(data)
        self.saveData(self.processedDataPath, self.processedDataName, data)
        return data

class BSRImages(DataSet):
    def __init__(self):
        super(BSRImages, self).__init__()
        self.fileExt = '.jpg'
        self.rootPath = '..\\Data\\BSR\\BSDS500\\data\\images\\'
        self.processedDataPath = self.rootPath + 'ProcessedData\\'
        self.processedDataName = 'processedImages.pkl'
        self.paths = [self.rootPath + i for i in ['train', 'test', 'val']]

    def readFile(self, file):
        datum = scipy.misc.imread(file)
        return datum

    def getData(self):
        data = []
        if isfile(self.processedDataPath + self.processedDataName):
            data = self.loadArray(self.processedDataPath, self.processedDataName)
            print('...loaded array of shape ' + str(data.shape))
            return data
        for set in self.paths:
            data.append(self.processImages(set, pad=True))
        data = np.vstack(data)
        self.saveData(self.processedDataPath, self.processedDataName, data)
        return data


def processBSR(dtype = np.uint8):
    labels = BSRLabels()
    images = BSRImages()
    x = images.getData()
    y = labels.getData()
    if x.dtype != dtype:
        x = x.astype(dtype)
    if y.dtype != dtype:
        y = x.astype(dtype)
    return x, y

def main():
    x, y = processBSR()


if __name__=='__main__':
    main()