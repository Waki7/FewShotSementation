import torch
import torch.nn as nn
import torch.nn.functional as F
from os.path import join, isfile
import DataProcessing.DataProcessor as data
import numpy as np
import Model.Config as cfg




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


class MetaLearner():
    def __init__(self, k = 5):
        self.k = 5
        self.model_path = '..\\StoredModels\\'
        self.model_name = 'MetaLearningModel.pk1'
        self.model_path = join(self.model_path, self.model_name)
        self.model = None
        self.load_model()

    def load_model(self):
        if isfile(self.model_path):
            self.model = torch.load(self.model_path)
            return True
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
            self.forward(meta_x, meta_y)

    def train_learner(self, learner_x, learner_y):
        '''
        Basically we are going to use x and y to give the meta learner its input paramters after every step
        :param learner_x:
        :param learner_y:
        :return:
        '''
        self.learner = Segmenter(lr=lr, downsample_ratio=4)

        optWhole = torch.optim.Adam(
            list(trainer.parameters()) + list(m.parameters()),
            # consider taking out trainer params if we want to include more gradient info in updates
            lr=lr, weight_decay=decay)

        criterion = torch.nn.BCELoss()

        n_train = x.shape[0]

        # potentially k fold on each batch in the future
        for i in range(0, len(n_train)):
            x = train_batches[i][:, 1:]
            y = train_batches[i][:, :1]
            for k in kshots:
                Y_out = m.forward(x)
                loss = criterion(input=Y_out, target=y)
                loss.backward(retain_graph=True)
                optWhole.zero_grad()

                for param in m.parameters():
                    h_t, i_t, f_t, c_t = trainer.forward(th_t1=param, dL_t=param.grad, L_t=loss)
                    # ^ figuire out how to initialize values before first pass for this shit, in the paper
                    param.data = trainer.update().detach()

            x = test_batches[i][:, 1:]
            y = test_batches[i][:, :1]
            Y_out = m.forward(x)
            loss = criterion(input=Y_out, target=y)
            loss.backward(retain_graph=True)
            optTrainer.step()
            optWhole.zero_grad()

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
    meta_learner.train_model()


if __name__ == "__main__":
    main()