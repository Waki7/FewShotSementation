import torch
import torch.nn as nn
import torch.nn.functional as F




class MetaLearner(nn.Module):
    def __init__(self, inputSize):
        numHU = 8
        bias = False

        self.wi = nn.Linear(in_features=inputSize, out_features=numHU, bias=bias)

        self.wf = nn.Linear(in_features=inputSize, out_features=numHU, bias=bias)


        self.wg = nn.Linear(in_features=inputSize, out_features=numHU, bias=bias)

        self.wo = nn.Linear(in_features=inputSize, out_features=numHU, bias=bias)


        self.wxi = nn.Linear(in_features=inputSize, out_features=numHU, bias=bias)
        self.wxi = nn.Linear(in_features=inputSize, out_features=numHU, bias=bias)


    def forward(self, h_t1, dL_t, L_t, th_t1, i_t1, f_t1, c_t1):
        x_t = torch.cat((dL_t, L_t, th_t1), dim=1)
        i_t = torch.sigmoid(self.wi(torch.cat((x_t, i_t1), dim=1)))
        f_t = torch.sigmoid(self.wf(torch.cat((x_t, f_t1), dim=1)))
        xh = torch.cat((x_t, h_t1), dim=1) #todo try splitting so that it is wx (x) + wh (h) instead of wxh(w;h)
        ci_t = torch.tanh(self.wg(xh))
        o_t = torch.sigmoid(self.wo(xh))
        c_t = torch.add(torch.dot(f_t, c_t1), torch.dot(i_t, ci_t))
        h_t = torch.dot(o_t, torch.tanh(c_t))
        return h_t, i_t, f_t, c_t # i_t is learning rate, ci_t is dL_t. c_t1= old parameters,
