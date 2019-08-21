import torch
import os
from os.path import join
import numpy as np
import random
from enum import Enum


class DataSetNames(Enum):
    BSR = 'BSR'
    VOC = 'VOC'


# pytorch
#################################################################################################################
use_cpu = False
device = torch.device('cpu') if use_cpu else torch.device('cuda')
dtype = torch.float32  # if use_cpu else torch.float32 #xentropy doesn't support float16
args = {'device': device, 'dtype': dtype}
torch.cuda.set_device(0)
#################################################################################################################

#### experiments
weights = False
#### experiments
model_size = 256
encoding_size = 1024
lr = .5
dataset_name = DataSetNames.VOC
experiment = 'FullSeg' + ('_Weights' if weights else '')

load_model = False
downsample_ratio = 4
encoding_downsample = 4
batch_size = 10  # roughly 45 for 64 model_size and half as u keep doubling
epochs = 10
np.random.seed(24)
random.seed(24)
torch.manual_seed(24)
torch.cuda.manual_seed(24)
torch.backends.cudnn.deterministic = True  # not gonna be deterministic.... https://github.com/pytorch/pytorch/issues/12207
torch.backends.cudnn.benchmark = False

# Paths
#################################################################################################################
experiment_path = '..\\ExperimentResults\\' + dataset_name.value + '\\' + experiment + '\\' + str(
    model_size) + '-' + str(encoding_size) + '\\'
print(experiment_path)
processed_data_path = '..\\Data\\ProcessedData\\'
stored_model_path = join(experiment_path, 'model_' + str(lr))
graph_file_path = join(experiment_path, 'heatmap_confusion_matrix' + str(lr))
#################################################################################################################


if not os.path.exists(experiment_path):
    os.makedirs(experiment_path)
console_file_name = 'console_' + str(lr) + '.txt'
console_file_path = join(experiment_path, console_file_name)
prnt = {'file': open(console_file_path, 'w'), 'flush': True} if not load_model else {}
