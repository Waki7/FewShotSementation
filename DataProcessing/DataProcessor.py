import scipy.io
import os
from os.path import join, isfile
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat



class BSRLabels():
    def __init__(self):
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

class BSRImages():
    def __init__(self):
        self.fileExt = '.jpg'
        self.rootPath = '..\\Data\\BSR\\BSDS500\\data\\images\\'
        self.paths = [self.rootPath + i for i in ['train', 'test', 'val']]

    def fileReader(self, file):
        datum = scipy.misc.imread(file)
        return datum

def process(directoryPath, dataSet): # should just get all files in directory as ndarray
    data = []
    for filename in os.listdir(directoryPath):
        if filename.endswith(dataSet.ext):
            samplePath = join(directoryPath, filename)
            data.append(dataSet.fileReader(samplePath))
            plt.imshow(data[0])
            plt.show()
            return

def processBSR():
    x = []
    y = []
    for set in ['train', 'test', 'val']: #do this specific splitting logic outside of this method
        process(BSR_path_images + set)
        process(BSR_path_labels + set)

def main():
    processBSR()


if __name__=='__main__':
    main()