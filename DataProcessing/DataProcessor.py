import scipy.io
import os
from os.path import join, isfile
import numpy as np
import cv2
import pickle
import Model.Config as cfg
import matplotlib.pyplot as plt
from scipy.io import loadmat
from torchvision import transforms

def get_experiment_data():
    switcher = {
        cfg.DataSetNames.VOC : DataVOC,
        cfg.DataSetNames.BSR : DataBSR,
    }
    return switcher.get(cfg.dataset_name)

def save_object(array, path, filename = None):
    if filename is not None:
        if not os.path.exists(path):
            os.makedirs(path)
        path = join(path, filename)
    with open(path, 'wb') as f:
        pickle.dump(array, f)


def load_object(path, filename = None):
    if filename is not None:
        path = join(path, filename)
    if isfile(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    else:
        return None

class ProcessedDataSet():
    def __init__(self, dataset_name, x_dtype=None, y_dtype=None, downsample_ratio = 1):
        self.x = None
        self.y = None
        self.dataset_name = dataset_name
        self.stored_file_name = self.dataset_name + '.pkl'
        self.n_classes = None
        self.x_shape = None
        self.x_dtype = x_dtype
        self.y_dtype = y_dtype
        self.class_weights = None
        self.data_images = DataFileLoader(downsample_ratio)
        self.data_labels = DataFileLoader(downsample_ratio)
        self.downsample_ratio = downsample_ratio

    def calc_class_weights(self, y):
        unique, counts = np.unique(y, return_counts=True)
        totalCount = sum(counts)
        self.class_weights = [totalCount / c for c in counts]

    def get_train_data(self):
        if self.x is None and self.y is None:
            self.load_data()
        if self.data_images.train_range is None:
            raise NotImplementedError
        indeces = self.data_images.train_range
        return self.x[indeces], self.y[indeces]

    def get_val_data(self):
        if self.x is None and self.y is None:
            self.load_data()
        if self.data_images.val_range is None:
            raise NotImplementedError
        indeces = self.data_images.val_range
        return self.x[indeces], self.y[indeces]

    def get_test_data(self):
        if self.x is None and self.y is None:
            self.load_data()
        if self.data_images.test_range is None:
            raise NotImplementedError
        indeces = self.data_images.test_range
        return self.x[indeces], self.y[indeces]

    def get_full_data(self):
        if self.x is None and self.y is None:
            self.load_data()
        return self.x, self.y

    def get_class_weights(self):
        return self.class_weights

    def try_load(self):
        if isfile(cfg.processed_data_path + self.stored_file_name):
            obj_dict = load_object(cfg.processed_data_path, self.stored_file_name)
            self.__dict__.update(obj_dict)
            return True
        return False

    def save(self):
        save_object(self.__dict__, cfg.processed_data_path, self.stored_file_name)

    def load_data(self):
        raise NotImplementedError

    def set_data_split(self, train_val_split, val_test_split):
        self.train_range = range(0, train_val_split)
        self.val_range = range(train_val_split, val_test_split)
        self.test_range = range(val_test_split, self.x_shape[0])

    def load_data(self):  # c x h x w
        if not self.try_load():
            x_full, x = self.data_images.load_data_from_files()
            y_full, y = self.data_labels.load_data_from_files()
            self.calc_class_weights(y_full)
            self.n_classes = len(self.class_weights)
            if self.x_dtype is not None and x.dtype != self.x_dtype:
                x = x.astype(self.x_dtype)
            if self.y_dtype is not None and y.dtype != self.y_dtype:
                y = y.astype(self.y_dtype)
            self.x = self.cleanInput(x)
            self.y = y
            self.x_shape = self.x.shape
            self.save()

    def cleanInput(self, x):
        print('...reshaped from ', x.shape)
        for i in range(x.shape[-1]):
            average = np.average(x[:, :, :, i])
            x[:, :, :, i] = (x[:, :, :, i] - average) / average
        if len(x.shape) > 3:
            x = np.transpose(x, (0, 3, 1, 2))
        print('to ', x.shape)
        return x


class DataBSR(ProcessedDataSet):
    def __init__(self, x_dtype=np.float32, y_dtype=np.float32, downsample_ratio=4):
        super(DataBSR, self).__init__('BSR', x_dtype, y_dtype)
        self.downsample_ratio = downsample_ratio
        self.data_images = BSRImages(downsample_ratio)
        self.data_labels = BSRLabels(downsample_ratio)

class DataVOC(ProcessedDataSet):
    def __init__(self, x_dtype=np.float32, y_dtype=np.float32, downsample_ratio=4):
        super(DataVOC, self).__init__('VOC', x_dtype, y_dtype)
        self.downsample_ratio = downsample_ratio
        self.data_images = VOCImages(downsample_ratio)
        self.data_labels = VOCLabels(downsample_ratio)

class DataFileLoader():
    def __init__(self, downsample_ratio):
        self.stored_data_path = None
        self.stored_file_name = None
        self.sampled_file_name = None
        self.downsample_ratio = downsample_ratio
        self.train_range = None
        self.val_range = None
        self.test_range = None

    def read_file(self, file):
        raise NotImplementedError

    def get_file_sets(self):
        '''
        :return: list of list of files, if applicable the first dimension should be split by train, val, test sets
        '''
        raise NotImplementedError

    def pad(self, image: np.ndarray, max_shape: list):
        pad_width = [(0, max_shape[i] - image.shape[i]) for i in range(len(max_shape))]
        return np.pad(image,
                      pad_width=pad_width, mode='constant')

    def shape_data_uniform(self, data: list): # todo bucketize if different sizes
        max_dims = [0]*len(data[0].shape)
        for i in range(0, len(data)):
            image = data[i]
            if image.shape[0] > image.shape[1]:  # want all images to be in portrait
                image = np.swapaxes(image, 0, 1)
            max_dims = [max(image.shape[dim_idx], max_dims[dim_idx]) for dim_idx in range(0, len(max_dims))]
        return [self.pad(image=image, max_shape=max_dims) for image in data]

    def read_files(self, files):
        data = []
        data_downsampled = []
        for sample_path in files:
            image, image_downsampled = self.read_file(sample_path)
            data.append(image)
            data_downsampled.append(image_downsampled)
        data = self.shape_data_uniform(data)
        data_downsampled = self.shape_data_uniform(data_downsampled)
        return np.stack(data), np.stack(data_downsampled)

    def load_data_from_files(self):
        data = []
        data_downsampled = []
        file_sets = self.get_file_sets()
        for file_list in file_sets:
            processed = self.read_files(file_list)
            data.append(processed[0])
            data_downsampled.append(processed[1])
        assert not any(np.any(np.isnan(data_i)) for data_i in data)
        data = np.vstack(data)
        data_downsampled = np.vstack(data_downsampled)
        if len(file_sets) is 3:
            train_val_split, val_test_split = len(file_sets[0]), len(file_sets[0]) + len(file_sets[1])
            self.train_range = range(0, train_val_split)
            self.val_range = range(train_val_split, val_test_split)
            self.test_range = range(val_test_split, data.shape[0])
        return data, data_downsampled


class VOCImages(DataFileLoader):
    def __init__(self, downsample_ratio):
        super(VOCImages, self).__init__(downsample_ratio)
        self.file_ext = '.jpg'
        self.root_path = '..\\Data\\VOC\\VOCdevkit\\VOC2012\\'
        self.images_path =  self.root_path + 'JPEGImages\\'
        self.image_set_paths = [self.root_path + 'ImageSets\\Segmentation\\' + i + '.txt' for i in ['train', 'trainval', 'val']]

    def read_file(self, file):
        datum = scipy.misc.imread(file)
        datum_downsampled = downsample(datum, ratio=self.downsample_ratio,
                                       interpolation=cv2.INTER_LINEAR)
        return datum, datum_downsampled

    def get_file_sets(self):
        file_sets = []
        for path in self.image_set_paths:
            file_names = []
            with open(path, 'r') as f:
                for line in f:
                    image_name_line = line.strip() + self.file_ext
                    sample_path = join(self.images_path, image_name_line)
                    file_names.append(sample_path)
            file_sets.append(file_names)
        return file_sets

class VOCLabels(DataFileLoader):
    def __init__(self, downsample_ratio):
        super(VOCLabels, self).__init__(downsample_ratio)
        self.file_ext = '.png'
        self.root_path = '..\\Data\\VOC\\VOCdevkit\\VOC2012\\'
        self.labels_path = self.root_path + 'SegmentationClass\\'
        self.image_set_paths = [self.root_path + 'ImageSets\\Segmentation\\' + i + '.txt' for i in ['train', 'trainval', 'val']]

    def read_file(self, file):
        datum = scipy.misc.imread(file)
        a, b = np.unique(return_counts=True)
        print(a)
        print(b)
        print(datum)
        print(datum.shape)
        plt.imshow(datum)
        plt.show()
        print(exit(9))
        datum_downsampled = downsample(datum, ratio=self.downsample_ratio,
                                       interpolation=cv2.INTER_LINEAR)
        return datum, datum_downsampled

    def get_file_sets(self):
        file_sets = []
        for path in self.image_set_paths:
            file_names = []
            with open(path, 'r') as f:
                for line in f:
                    image_name_line = line.strip() + self.file_ext
                    sample_path = join(self.labels_path, image_name_line)
                    file_names.append(sample_path)
            file_sets.append(file_names)
        return file_sets


class BSRImages(DataFileLoader):
    def __init__(self, downsample_ratio):
        super(BSRImages, self).__init__(downsample_ratio)
        self.file_ext = '.jpg'
        self.root_path = '..\\Data\\BSR\\BSDS500\\data\\images\\'
        self.image_set_paths = [self.root_path + i for i in ['train', 'val', 'test']]

    def read_file(self, file):
        datum = scipy.misc.imread(file)
        datum_downsampled = downsample(datum, ratio=self.downsample_ratio,
                                       interpolation=cv2.INTER_LINEAR)
        return datum, datum_downsampled

    def get_file_sets(self):
        file_sets = []
        for path in self.image_set_paths:
            file_names = []
            for filename in os.listdir(path):
                if filename.endswith(self.file_ext):
                    sample_path = join(path, filename)
                    file_names.append(sample_path)
            file_sets.append(file_names)
        return file_sets


class BSRLabels(DataFileLoader):
    def __init__(self, downsample_ratio):
        super(BSRLabels, self).__init__(downsample_ratio)
        self.file_ext = '.mat'
        self.root_path = '..\\Data\\BSR\\BSDS500\\data\\groundTruth\\'
        self.image_set_paths = [self.root_path + i for i in ['train', 'val', 'test']]
        self.mat_key = 'groundTruth'
        self.segmentation_index = 0
        self.boundary_index = 1

    def read_file(self, file):
        mat = scipy.io.loadmat(file)
        mat_data = np.squeeze(mat[self.mat_key][0, 0]).item(0)
        datum = mat_data[self.segmentation_index]  # segementation ground truth, mat_data[1] is the boundary boxes
        datum = datum - 1
        datum_downsampled = downsample(datum, ratio=self.downsample_ratio,
                                       interpolation=cv2.INTER_NEAREST)
        return datum, datum_downsampled

    def get_file_sets(self):
        file_sets = []
        for path in self.image_set_paths:
            file_names = []
            for filename in os.listdir(path):
                if filename.endswith(self.file_ext):
                    sample_path = join(path, filename)
                    file_names.append(sample_path)
            file_sets.append(file_names)
        return file_sets

class KShotSegmentation():
    def __init__(self, x=None, y=None, k=5, downsample_ratio = 2):
        self.folder_path = '..\\Data\\MetaLearnerData\\'
        self.file_name = 'BSR_meta_data'+str(k)+'.pkl'
        self.stored_path = join(self.folder_path, self.file_name)
        self.meta_data = load_object(self.folder_path, self.file_name)
        self.downsample_ratio = downsample_ratio
        if self.meta_data is None:
            if x is None and y is None:
                data = DataBSR(x_dtype=np.float32, y_dtype=np.int32)
                data.load_data()
                x, y = data.get_full_data()
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
        save_object([meta_xs, meta_ys], self.folder_path, self.file_name)

    def get_dataset(self, idx):
        return self.meta_xs[idx], self.meta_ys[idx]



def downsample(img, ratio, interpolation=cv2.INTER_NEAREST):
    new_h = img.shape[0] // ratio
    new_w = img.shape[1] // ratio
    img = cv2.resize(img, (new_w, new_h), interpolation=interpolation) #opencv takes w x h instead of h x w in numpy
    return img
