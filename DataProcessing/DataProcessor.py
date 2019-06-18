import scipy.io
import os
from os.path import join, isfile
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat


class DataSet():
    def __init__(self):
        self.fileExt = None

    def fileReader(self, file):
        raise NotImplementedError

    def process(self, directoryPath):  # should just get all files in directory as ndarray
        data = []
        for filename in os.listdir(directoryPath):
            if filename.endswith(self.fileExt):
                samplePath = join(directoryPath, filename)
                data.append(self.fileReader(samplePath))
        return np.ndarray(data)

class BSRLabels(DataSet):
    def __init__(self):
        super(BSRLabels, self).__init__()
        self.fileExt = '.mat'
        self.rootPath = '..\\Data\\BSR\\BSDS500\\data\\groundTruth\\'
        self.paths = [self.rootPath + i for i in ['train', 'test', 'val']]
        self.matKey = 'groundTruth'
        self.segmentationIndex = 0
        self.boundaryIndex = 1

    def fileReader(self, file):
        mat = scipy.io.loadmat(file)
        mat_data = np.squeeze(mat[self.matKey][0, 0]).item(0)
        datum = mat_data[self.segmentationIndex] #segementation ground truth, mat_data[1] is the boundary boxes
        return datum

    def getData(self):
        for set in self.paths:
            self.process(set)

class BSRImages(DataSet):
    def __init__(self):
        super(BSRImages, self).__init__()
        self.fileExt = '.jpg'
        self.rootPath = '..\\Data\\BSR\\BSDS500\\data\\images\\'
        self.paths = [self.rootPath + i for i in ['train', 'test', 'val']]

    def fileReader(self, file):
        datum = scipy.misc.imread(file)
        return datum

    def getData(self):
        for set in self.paths:
            self.process(set)


def processBSR():
    labels = BSRLabels()
    images = BSRImages()
    x = labels.getData()
    y = images.getData()
    return x, y

def main():
    x, y = processBSR()
    print(x.shape)
    print(y.shape)


if __name__=='__main__':
    main()