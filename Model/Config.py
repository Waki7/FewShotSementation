import torch
import os
import sys
import numpy as np
import random
from os.path import join

#pytorch
#################################################################################################################
use_cpu = False
device = torch.device('cpu') if use_cpu else torch.device('cuda')
dtype = torch.float32  # if use_cpu else torch.float32 #xentropy doesn't support float16
args = {'device': device, 'dtype': dtype}
#################################################################################################################



load = False
batch_size = 45 #roughly 45 for 64 model_size and half as u keep doubling
model_size = 64
epochs = 1000
lr = .075
np.random.seed(24)
random.seed(24)
torch.manual_seed(24)
torch.cuda.manual_seed(24)
torch.backends.cudnn.deterministic = True # not gonna be deterministic.... https://github.com/pytorch/pytorch/issues/12207
torch.backends.cudnn.benchmark=False

#Paths
#################################################################################################################
experiment_path = '..\\ExperimentResults\\BSR\\FullSeg\\ClassWeights\\' + str(model_size) + '\\'
processed_data_path = '..\\Data\\ProcessedData\\'
stored_model_path = join(experiment_path, 'FullBSRSegmenter'+str(lr))
graph_file_path = join(experiment_path, 'heatmap_confusion_matrix' + str(lr))
#################################################################################################################


if not os.path.exists(experiment_path):
    os.makedirs(experiment_path)
console_file_name = 'console' + str(lr) + '.txt'
console_file_path = join(experiment_path, console_file_name)
prnt = {'file': open(console_file_path, 'w'), 'flush': True} if not load else {}
