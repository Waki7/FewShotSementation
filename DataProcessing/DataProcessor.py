import scipy.io
import os
from os.path import join, isfile
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

BSR_path_labels = '..\\Data\\BSR\\BSDS500\\data\\groundTruth\\train' # 0
BSR_path_images = '..\\Data\\BSR\\BSDS500\\data\\images\\train'
BSRTestKey = 'groundTruth'

def process(directoryPath, dataExt = '.jpg'): # should just get all files in directory as ndarray
    data = []
    for set in ['train', 'test', 'val']: #do this specific splitting logic outside of this method
        for filename in os.listdir(directoryPath):
            if filename.endswith(dataExt):
                samplePath = join(directoryPath, filename)

                mat = scipy.io.loadmat(samplePath)
                mat_data = np.squeeze(mat[BSRTestKey][0,0]).item(0)
                segmentationGroundTruth = mat_data[0]
                #mat[1] is the boundaries
                print(segmentationGroundTruth.shape)

                plt.imshow(segmentationGroundTruth)
                plt.show()
                plt.show()

                print(exit(9))
def main():
    process(BSR_path)


if __name__=='__main__':
    main()