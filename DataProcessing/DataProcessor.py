import scipy.io
import os
from os.path import join, isfile
import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt
from scipy.io import loadmat
from torchvision import transforms


def saveData(array, path, filename):
    if not os.path.exists(path):
        os.makedirs(path)
    path = join(path, filename)
    with open(path, 'wb') as f:
        pickle.dump(array, f)


def loadArray(path, filename):
    if filename is not None:
        path = join(path, filename)
    if isfile(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    else:
        return None

class ProcessedDataSet():
    def __init__(self, x_dtype=None, y_dtype=None):
        self.x = None
        self.y = None
        self.n_classes = None
        self.x_dtype = x_dtype
        self.y_dtype = y_dtype
        self.class_weights = None
        self.data_labels = DataSet()
        self.data_images = DataSet()

    def calc_class_weights(self, y):
        unique, counts = np.unique(y, return_counts=True)
        totalCount = sum(counts)
        self.class_weights = [totalCount / c for c in counts]

    def get_train_data(self):
        if self.x is None and self.y is None:
            self.load_data()
        if len(self.x) is not 3:
            raise NotImplementedError
        indeces = range(self.data_images.train_range)
        return self.x[indeces], self.y[indeces]

    def get_validation_data(self):
        if self.x is None and self.y is None:
            self.load_data()
        if len(self.x) is not 3:
            raise NotImplementedError
        indeces = range(self.data_images.validate_range)
        return self.x[indeces], self.y[indeces]

    def get_test_data(self):
        if self.x is None and self.y is None:
            self.load_data()
        if len(self.x) is not 3:
            raise NotImplementedError
        indeces = range(self.data_images.test_range)
        return self.x[indeces], self.y[indeces]

    def get_full_data(self):
        if self.x is None and self.y is None:
            self.load_data()
        # print(np.shape(self.x))
        # print(np.shape(self.y))
        # print(np.shape(np.concatenate(self.x, axis=1)))
        # print(np.shape(np.concatenate(self.y, axis=1)))
        # return np.concatenate(self.x, axis=1), np.concatenate(self.y, axis=1)
        return self.x, self.y

    def get_class_weights(self):
        return self.class_weights

    def load_data(self):
        raise NotImplementedError

class DataBSR(ProcessedDataSet):
    def __init__(self, x_dtype=np.float32, y_dtype=np.float32, downsample_ratio=4):
        super(DataBSR, self).__init__(x_dtype, y_dtype)
        self.downsample_ratio = downsample_ratio
        self.data_images = BSRImages(self.downsample_ratio)
        self.data_labels = BSRLabels(self.downsample_ratio)

    def load_data(self):  # c x h x w
        x_full, x = self.data_images.get_data()
        y_full, y = self.data_labels.get_data()
        self.calc_class_weights(y_full[0])
        self.n_classes = len(self.class_weights)
        if self.x_dtype is not None and x[0].dtype != self.x_dtype:
            x = [xi.astype(self.x_dtype) for xi in x]
        if self.y_dtype is not None and y[0].dtype != self.y_dtype:
            y = [yi.astype(self.y_dtype) for yi in y]
        self.x = [cleanInput(xi) for xi in x]
        self.y = y

class DataSet():
    def __init__(self):
        self.file_ext = None
        self.root_path = None
        self.stored_data_path = None
        self.stored_file_name = None
        self.sampled_file_name = None
        self.paths = None
        self.downsample_ratio = 1
        self.train_range = None
        self.test_range = None
        self.validate_range = None

    def read_file(self, file):
        raise NotImplementedError

    def pad(self, image: np.ndarray, maxShape):
        pad_width = [(0, maxShape[i] - image.shape[i]) for i in range(len(maxShape))]
        return np.pad(image,
                      pad_width=pad_width, mode='constant')


    def process_images(self, directoryPath):
        data = []
        data_downsampled = []
        for filename in os.listdir(directoryPath):
            if filename.endswith(self.file_ext):
                samplePath = join(directoryPath, filename)
                image, image_downsampled = self.read_file(samplePath)
                if image.shape[0] > image.shape[1]:  # want all images to be in portrait
                    image = np.swapaxes(image, 0, 1)
                    image_downsampled = np.swapaxes(image_downsampled, 0, 1)
                data.append(image)
                data_downsampled.append(image_downsampled)
        return np.stack(data), np.stack(data_downsampled)

    def get_data(self):
        data = []
        data_downsampled = []
        if isfile(self.stored_data_path + self.stored_file_name) and isfile(
                self.stored_data_path + self.sampled_file_name):
            data = loadArray(self.stored_data_path, self.stored_file_name)
            data_downsampled = loadArray(self.stored_data_path, self.sampled_file_name)
            print('...loaded arrays of shape ' + str(np.shape(data)) + ' and downsampled to ' +
                  str(np.shape(data_downsampled)))
            return data, data_downsampled
        for set in self.paths:
            processed = self.process_images(set)
            data.append(processed[0])
            data_downsampled.append(processed[1])
        assert not any(np.any(np.isnan(data_i)) for data_i in data)
        data = np.vstack(data)
        data_downsampled = np.vstack(data_downsampled)
        saveData(data, self.stored_data_path, self.stored_file_name)
        saveData(data_downsampled, self.stored_data_path, self.sampled_file_name)
        return data, data_downsampled


class VOCImages(DataSet):
    def __init__(self, downsample_ratio):
        super(VOCImages, self).__init__()
        self.file_ext = '.jpg'
        self.root_path = '..\\Data\\VOC\\VOC2012\\JPEGImages\\'



class VOCLabels(DataSet):
    def __init__(self, downsample_ratio):
        super(VOCLabels, self).__init__()
        self.file_ext = '.png'
        self.root_path = '..\\Data\\VOC\\VOC2012\\SegmentationClass\\'



    def read_file(self):
        pass


class BSRImages(DataSet):
    def __init__(self, downsample_ratio):
        super(BSRImages, self).__init__()
        self.file_ext = '.jpg'
        self.root_path = '..\\Data\\BSR\\BSDS500\\data\\images\\'
        self.stored_data_path = self.root_path + 'ProcessedData\\'
        self.stored_file_name = 'processedImages.pkl'
        self.sampled_file_name = 'downsampledImages.pkl'
        self.paths = [self.root_path + i for i in ['train', 'test', 'val']]
        self.downsample_ratio = downsample_ratio

    def read_file(self, file):
        datum = scipy.misc.imread(file)
        datum_downsampled = downsample(datum, ratio=self.downsample_ratio,
                                       interpolation=cv2.INTER_LINEAR)
        return datum, datum_downsampled


class BSRLabels(DataSet):
    def __init__(self, downsample_ratio):
        super(BSRLabels, self).__init__()
        self.file_ext = '.mat'
        self.root_path = '..\\Data\\BSR\\BSDS500\\data\\groundTruth\\'
        self.stored_data_path = self.root_path + 'ProcessedData\\'
        self.stored_file_name = 'processedLabels.pkl'
        self.sampled_file_name = 'downsampledLabels.pkl'
        self.paths = [self.root_path + i for i in ['train', 'test', 'val']]
        self.mat_key = 'groundTruth'
        self.segmentation_index = 0
        self.boundary_index = 1
        self.downsample_ratio = downsample_ratio

    def read_file(self, file):
        mat = scipy.io.loadmat(file)
        mat_data = np.squeeze(mat[self.mat_key][0, 0]).item(0)
        datum = mat_data[self.segmentation_index]  # segementation ground truth, mat_data[1] is the boundary boxes
        datum = datum - 1
        datum_downsampled = downsample(datum, ratio=self.downsample_ratio,
                                       interpolation=cv2.INTER_NEAREST)
        # datum1 = mat_data[1]
        # plt.imshow(datum)
        # plt.show()
        return datum, datum_downsampled


class KShotSegmentation():
    def __init__(self, x=None, y=None, k=5, downsample_ratio = 2):
        self.folder_path = '..\\Data\\MetaLearnerData\\'
        self.file_name = 'BSR_meta_data'+str(k)+'.pkl'
        self.stored_path = join(self.folder_path, self.file_name)
        self.meta_data = loadArray(self.folder_path, self.file_name)
        self.downsample_ratio = downsample_ratio
        if self.meta_data is None:
            if x is None and y is None:
                data = DataBSR(x_dtype=np.float32, y_dtype=np.int32)
                x, y = data.load_data()
            self.make_data(x, y, k)
        else:
            self.meta_xs = self.meta_data[0]
            self.meta_ys = self.meta_data[1]


    def make_data(self, x, y, k):
        meta_xs = []
        meta_ys = []

        for meta_idx in range(0, x.shape[0]):
            target_x = x[meta_idx]
            target_y = y[meta_idx]
            labels = np.unique(target_y, return_counts=False)
            ks = [k]*len(labels)
            meta_ys.append((target_x, target_y))
            meta_x = []
            for label_idx in range(len(labels)):
                label = labels[label_idx]
                ks[label_idx] -= np.sum([label in img[1] for img in meta_x])
                for data_idx in range(0, x.shape[0]):
                    if data_idx != meta_idx:
                        if ks[label_idx] <= 0:
                            break
                        data_x = x[data_idx]
                        data_y = y[data_idx]
                        if label in data_y:
                            ks[label_idx] -= 1
                            meta_x.append((data_x, data_y))
            meta_xs.append(meta_x)
        self.meta_xs = meta_xs
        self.meta_ys = meta_ys
        saveData([meta_xs, meta_ys], self.folder_path, self.file_name)

    def get_dataset(self, idx):
        return self.meta_xs[idx], self.meta_ys[idx]



def downsample(img, ratio, interpolation=cv2.INTER_NEAREST):
    new_h = img.shape[0] // ratio
    new_w = img.shape[1] // ratio
    img = cv2.resize(img, (new_w, new_h), interpolation=interpolation) #opencv takes w x h instead of h x w in numpy
    return img


def cleanInput(x):
    print('...reshaped from ', x.shape)
    channel_averages = [np.average(x[:, :, i]) for i in range(x.shape[-1])]
    for i in range(x.shape[-1]):
        average = np.average(x[:, :, :, i])
        x[:, :, :, i] = (x[:, :, :, i] - average) / average
        # print(average)
        # print(x[:,:,:,i].shape)
        # print(np.average(x[:,:,:,i]))
    if len(x.shape) > 3:
        x = np.transpose(x, (0, 3, 1, 2))
        # x = x / np.max(x)  # scale [0,255] -> [0,1]
    print('to ', x.shape)
    return x

