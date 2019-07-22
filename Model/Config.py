import torch
import os
from os.path import join

use_cpu = False
device = torch.device('cpu') if use_cpu else torch.device('cuda')
dtype = torch.float32  # if use_cpu else torch.float32 #xentropy doesn't support float16
args = {'device': device, 'dtype': dtype}
model_size = 256
lr = .01
experiment_path = '..\\ExperimentResults\\BSR\\FullSeg\\v1\\' + str(model_size) + '\\'
if not os.path.exists(experiment_path):
    os.makedirs(experiment_path)
result_file_name = 'console'+str(lr)
result_file_path = join(experiment_path, result_file_name)
prnt = {'file': open(result_file_path, 'w'), 'flush': True}
