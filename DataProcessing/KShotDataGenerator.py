import DataProcessing.DataProcessor as dp
from os.path import join, isfile
import numpy as np
import Model.Config as cfg

class KShotSegmentationDataGenerator(dp.ProcessedDataSet):
    def __init__(self, dataset: dp.ProcessedDataSet, n_samples=None, k=5):
        super(KShotSegmentationDataGenerator, self).__init__('{}_{}-shot'.format(dataset.dataset_name, k))
        self.folder_path = '..\\Data\\MetaLearnerData\\'
        self.dataset = dataset
        if self.dataset.n_samples is None:
            raise ValueError('data is not loaded')
        if n_samples is None:
            self.n_samples = len(self.dataset.test_range)

        self.k = k
        self.meta_x_indeces = None
        self.meta_y_indeces = None

    def save(self):
        state = self.__dict__.copy()
        del state['dataset']
        dp.save_object(state, cfg.processed_data_path, self.stored_file_name)

    def create_meta_sets(self):  # c x h x w
        meta_x_indeces = []
        meta_y_indeces = []
        trainee_x = self.dataset.x
        trainee_y = self.dataset.y
        for meta_idx in range(0, self.n_samples):

            labels = np.unique(trainee_y[meta_idx], return_counts=False)
            labels = labels[labels != self.dataset.ignore_index]
            meta_y_indeces.append(meta_idx)

            meta_x = []
            for label_idx in range(len(labels)):
                label = labels[label_idx]
                k = self.k
                for data_idx in range(0, self.dataset.x.shape[0]):
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
