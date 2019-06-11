import torch
import torch.nn as nn
import torch.nn.functional as F




class LifeNetwork(nn.Module):
    def __init__(self, totalWeights):
        super(LifeNetwork, self).__init__()

        self.agent = agent
        numS = 8
        bias = False
        l1_out_features = 0
        self.wl1 = {}
        for i in range(0, len(self.agent.inputChannels)):
            inputChannel = self.agent.inputChannels[i]
            self.wl1[inputChannel] = nn.Linear(
                in_features=self.agent.inputShpes[i], out_features=numS, bias=bias).cuda()
            l1_out_features += self.wl1[inputChannel].out_features

        self.wl2 = nn.Linear(
            in_features=l1_out_features, out_features=numS, bias=bias)

        self.wly = {}
        for i in range(0, len(self.agent.outputChannels)):
            outputChannel = self.agent.outputChannels[i]
            self.wly[outputChannel] = nn.Linear(
                in_features=self.wl2.out_features, out_features=self.agent.outputShpes[i], bias=bias).cuda()


    def forward(self, envInput, selfInput):
        l0, l1, l1_cmbn = {}, {}, []
        input = torch.cat((envInput, selfInput), dim=1)

def main():
    pass

if __name__ == "__main__":
    main()