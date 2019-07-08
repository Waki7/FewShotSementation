import torch
import torch.nn as nn
import torch.nn.functional as F
from os.path import join, isfile



class MetaLearner():
    def __init__(self):
        self.model_path = '..\\StoredModels\\'
        self.model_name = 'MetaLearningModel.pk1'
        self.model_path = join(self.model_path, self.model_name)
        self.model = None
        self.try_load_model()

    def try_load_model(self):
        if isfile(self.model_path):
            self.model = MetaLearningModel()
            self.model.load_state_dict(torch.load(self.model_path))

    def get_optimizer(self, parameters):
        if self.model is None:
            self.train_model()
        self.set_learner(parameters)
        return self.model

    def train_model(self):
        pass

    def set_learner(self, parameters):
        self.parameters = parameters

    def step(self, loss):
        for param in self.parameters:
            weight_update = self.forward(param, param.grad.data, loss)
            param.data = self.model.update().detach()

    def zero_grad(self):
        pass


class MetaLearningModel(nn.Module):
    def __init__(self, inputSize = 1):
        numHU = 8
        bias = False

        self.wi = nn.Linear(in_features=inputSize, out_features=numHU, bias=bias)

        self.wf = nn.Linear(in_features=inputSize, out_features=numHU, bias=bias)


        self.wg = nn.Linear(in_features=inputSize, out_features=numHU, bias=bias)

        self.wo = nn.Linear(in_features=inputSize, out_features=numHU, bias=bias)


        self.wxi = nn.Linear(in_features=inputSize, out_features=numHU, bias=bias)
        self.wxi = nn.Linear(in_features=inputSize, out_features=numHU, bias=bias)

    def forward(self, th_t1, dL_t, L_t):
        x_t = torch.cat((dL_t, L_t, th_t1), dim=1)
        i_t = torch.sigmoid(self.wi(torch.cat((x_t, self.i_t1), dim=1)))
        f_t = torch.sigmoid(self.wf(torch.cat((x_t, self.f_t1), dim=1)))
        xh = torch.cat((x_t, self.h_t1), dim=1) #todo try splitting so that it is wx (x) + wh (h) instead of wxh(w;h)
        ci_t = torch.tanh(self.wg(xh))
        o_t = torch.sigmoid(self.wo(xh))
        c_t = torch.add(torch.dot(f_t, self.c_t1), torch.dot(i_t, ci_t))
        h_t = torch.dot(o_t, torch.tanh(c_t))

        self.h_t1 = h_t
        self.i_t1 = i_t
        self.f_t1 = f_t
        self.c_t1 = c_t

        self.lr_ = i_t
        self.th_t1_c = f_t
        self.dL_t = ci_t
        self.th_t1 = self.c_t1
        return h_t, i_t, f_t, c_t # i_t is learning rate, ci_t is -dL_t. c_t1= old parameters,

    def update(self):
        return torch.sum(torch.dot(self.th_t1_c, self.th_t1), torch.dot(self.lr_, self.dL_t))
