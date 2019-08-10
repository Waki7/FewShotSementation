import DataProcessing.DataProcessor as dp
from os.path import join, isfile
import numpy as np
import Model.Config as cfg

class KShotSegmentationDataGenerator(dp.ProcessedDataSet):
    def __init__(self, data_set: dp.ProcessedDataSet, n_samples=None, k=5):
        super(KShotSegmentationDataGenerator, self).__init__('{}_{}-shot'.format(data_set.dataset_name, k))
        self.folder_path = '..\\Data\\MetaLearnerData\\'
        self.data_set = data_set
        if self.data_set.n_samples is None:
            raise ValueError('data is not loaded')
        if n_samples is None:
            self.n_samples = len(self.data_set.test_range)

        self.k = k
        self.meta_x_indeces = None
        self.meta_y_indeces = None

    def save(self):
        state = self.__dict__.copy()
        del state['data_set']
        dp.save_object(state, cfg.processed_data_path, self.stored_file_name)

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
        print(np.shape(self.meta_x_indeces))

    def load_data(self):  # c x h x w
        if not self.try_load():
            self.create_meta_sets()
            self.save()
