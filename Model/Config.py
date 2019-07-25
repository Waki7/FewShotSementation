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
batch_size = 90 #roughly 45 for 64 model_size and half as u keep doubling
model_size = 32
epochs = 500
lr = .1
np.random.seed(24)
random.seed(24)
torch.manual_seed(24)
torch.cuda.manual_seed(24)
torch.backends.cudnn.deterministic = True

#Paths
#################################################################################################################
experiment_path = '..\\ExperimentResults\\BSR\\FullSeg\\v1\\' + str(model_size) + '\\'
processed_data_path = '..\\Data\\ProcessedData\\'
#################################################################################################################


if not os.path.exists(experiment_path):
    os.makedirs(experiment_path)
result_file_name = 'console'+str(lr)+'.txt'
result_file_path = join(experiment_path, result_file_name)
prnt = {'file': open(result_file_path, 'w'), 'flush': True} if not load else {}
