import DataProcessing.DataProcessor as dp
from os.path import join, isfile
import numpy as np
import Model.Config as cfg

class KShotSegmentationDataGenerator(dp.ProcessedDataSet):
    def __init__(self, data_set: dp.ProcessedDataSet, n_samples=None, k=5):
        super(KShotSegmentationDataGenerator, self).__init__('{}_{}-shot'.format(data_set.dataset_name, k))
        self.folder_path = '..\\Data\\MetaLearnerData\\'
        self.data_set = data_set
        if n_samples is None:
            self.n_samples = self.data_set.test_range[1] = self.data_set.test_range[0]

        self.k = k

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['data_set']
        return state

    def create_meta_sets(self):  # c x h x w
        meta_x_indeces = []
        meta_y_indeces = []
        trainee_x = self.data_set.x
        trainee_y = self.data_set.y
        for meta_idx in range(0, self.n_samples):

            labels = np.unique(trainee_y[meta_idx], return_counts=False)

            meta_y_indeces.append(meta_idx)

            meta_x = []
            k = self.k
            for label_idx in range(len(labels)):
                label = labels[label_idx]
                for data_idx in range(0, self.data_set.x.shape[0]):
                    if data_idx != meta_idx:
                        if k <= 0:
                            break
                        data_y = trainee_y[data_idx]
                        if label in data_y:
                            k -= 1
                            meta_x.append(data_idx)
            meta_x_indeces.append(meta_x)
        self.meta_x_indeces = meta_x_indeces
        self.meta_y_indeces = meta_y_indeces
        dp.save_object([meta_x_indeces, meta_y_indeces], self.folder_path, self.file_name)

    def load_data(self):  # c x h x w
        if not self.try_load():
            self.create_meta_sets()

            ###3##totodoododtdodo todo jalksd
            x_full, x = self.data_observations.load_data_from_files()
            y_full, y = self.data_labels.load_data_from_files()
            self.n_samples = x_full.shape[0]
            self.split_data()
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
