import torch

use_cpu = False
device = torch.device('cpu') if use_cpu else torch.device('cuda')
dtype = torch.float32  # if use_cpu else torch.float32 #xentropy doesn't support float16
args = {'device': device, 'dtype': dtype}
model_size = 128
result_file_name = 'console'+str(model_size)+'.txt'
prnt = {'file': open(result_file_name, 'w'), 'flush': True}
