import torch
import torch.nn as nn
import torch.nn.functional as F
from os.path import join, isfile
import DataProcessing.DataProcessor as data
import Model.SegmentationModel as seg
import numpy as np
import Model.Config as cfg




class MetaLearningModel(nn.Module):
    def __init__(self, input_size = 1):
        super(MetaLearningModel, self).__init__()

        numHU = 8
        bias = False
        states_initialized = False
        self.input_size = input_size

        self.wi = nn.Linear(in_features=input_size, out_features=numHU, bias=bias)

        self.wf = nn.Linear(in_features=input_size, out_features=numHU, bias=bias)


        self.wg = nn.Linear(in_features=input_size, out_features=numHU, bias=bias)

        self.wo = nn.Linear(in_features=input_size, out_features=numHU, bias=bias)


        self.wxi = nn.Linear(in_features=input_size, out_features=numHU, bias=bias)
        self.wxi = nn.Linear(in_features=input_size, out_features=numHU, bias=bias)

    def initalize_states(self, th_t1):  # initialize bias to be higher for forget gate i believe and lwoer for other
        self.i_t1 = torch.ones(self.input_size)
        self.f_t1 = torch.ones(
            self.input_size)  # i believe this at 1 means assume we don't forget any of original paramter
        self.c_t1 = th_t1
        self.h_t1 = torch.zeros(self.input_size)  # shape like output of o_t, torch.tanh(c_t)
        self.states_initialized = True

    def forward(self, th_t1, dL_t, L_t):
        if not self.states_initialized:
            self.initalize_states(th_t1)
        x_t = torch.cat((dL_t, L_t, th_t1), dim=1)
        i_t = torch.sigmoid(self.wi(torch.cat((x_t, self.i_t1), dim=1)))
        f_t = torch.sigmoid(self.wf(torch.cat((x_t, self.f_t1), dim=1)))

        xh = torch.cat((x_t, self.h_t1), dim=1) #todo try splitting so that it is wx (x) + wh (h) instead of wxh(w;h)
        ci_t = torch.tanh(self.wg(xh))
        o_t = torch.sigmoid(self.wo(xh))
        c_t = torch.add(torch.dot(f_t, self.c_t1), torch.dot(i_t, ci_t))
        h_t = torch.dot(o_t, torch.tanh(c_t))
        print(h_t.shape)
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


class MetaLearner():
    def __init__(self, k = 5, lr = .01, decay = .00001):
        self.k = 5
        self.lr = lr
        self.decay = decay
        self.model_path = '..\\StoredModels\\'
        self.model_name = 'MetaLearningModel.pk1'
        self.model_path = join(self.model_path, self.model_name)
        self.model = None
        self.load_model()

    def load_model(self):
        if isfile(self.model_path):
            self.model = torch.load(self.model_path)
            return True
        self.model = MetaLearningModel()
        return False

    def get_optimizer(self, parameters):
        if self.model is None:
            self.train_meta_learner()
        self.set_learner(parameters)
        return self.model

    def train_meta_learner(self):
        n_train = 5
        for i in range(0, n_train):
            meta_x, meta_y = self.data.get_dataset(i)
            self.meta_forward(meta_x, meta_y)



    def meta_forward(self, meta_x, meta_y):
        '''
        Basically we are going to use x and y to give the meta learner its input paramters after every step
        :param meta_x:
        :param meta_y:
        :return:
        '''
        unique = np.unique(meta_y[1])
        learner = seg.SegmentationModel(np.expand_dims(meta_x[0][0], axis=0).shape, len(unique))
        for learner_x, learner_y in meta_x:
            opt_full = torch.optim.Adam(
                list(learner.parameters()) + list(self.model.parameters()),
                # consider taking out trainer params if we want to include more gradient info in updates
                lr=self.lr, weight_decay=self.decay)
            criterion = torch.nn.BCELoss()
            criterion = self.train_learner(learner, opt_full, criterion, learner_x, learner_y)


        meta_y_x = meta_y[0]
        meta_y_y = meta_y[1]
        Y_out = learner.forward(torch.Tensor(meta_y_x).to(**cfg.args))
        loss = criterion(input=Y_out, target=torch.Tensor(meta_y_y).to(cfg.device).long())
        loss.backward()
        opt_full.step()
        opt_full.zero_grad()


    def train_learner(self, learner, opt, criterion, x, y):
        x, y = torch.Tensor(x).to(**cfg.args), torch.Tensor(y).to(cfg.device).long()
        x, y = torch.unsqueeze(x, dim=0), torch.unsqueeze(y, dim=0)
        Y_out = learner.forward(x)
        loss = criterion(input=Y_out, target=y)
        loss.backward(retain_graph=True)
        opt.zero_grad()

        for param in learner.parameters():
            h_t, i_t, f_t, c_t = self.model.forward(th_t1=param, dL_t=param.grad, L_t=loss)
            # ^ figuire out how to initialize values before first pass for this shit, in the paper
            param.data = self.model.update().detach()

    def set_learner(self, parameters):
        self.parameters = parameters

    def step(self, loss):
        for param in self.parameters:
            weight_update = self.forward(param, param.grad.data, loss)
            param.data = self.model.update().detach()

    def zero_grad(self):
        pass

    def load_data(self):
        self.data = data.KShotSegmentation(k=self.k)

def main():
    meta_learner = MetaLearner()
    meta_learner.load_data()
    meta_learner.train_meta_learner()


if __name__ == "__main__":
    main()
