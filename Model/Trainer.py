import Model.MetaLearner as ml
import Model.Segmenter as seg
import torch
from typing import List
import math
import numpy as np

def getDataSets():
    pass

def getTrainer():
    return ml.MetaLearner()

def trainTrainer(data: np.ndarray, trainer: ml.MetaLearner, models: List, kshots = 20, lr = .0005, decay = .005):
    """
    :param kshots: if data can't fit on gpu then increase number of kshots, implemented so that whole batch is ran on gpu

    :return:
    """
    optTrainer = torch.optim.Adam(trainer.parameters(), lr=lr, weight_decay=decay)
    for m in models:
        optWhole = torch.optim.Adam(
            list(trainer.parameters()) + list(m.parameters()), #consider taking out trainer params if we want to include more gradient info in updates
            lr=lr, weight_decay=decay)

        criterion = torch.nn.BCELoss()

        x_full = data.getX()
        n_train = x.shape[0]
        y_full = data.getY()

        n = int(math.ceil(len(data) / kshots))
        kshots = [data[i:min(i + n, len(data))] for i in range(0, len(data), n)]
        train_batches = [b[:-len(b)//2] for b in kshots]
        test_batches = [b[len(b)//2:] for b in kshots]

        # potentially k fold on each batch in the future
        for i in range(0, len(train_batches)):
            x = train_batches[i][:,1:]
            y = train_batches[i][:,:1]
            for k in kshots:
                Y_out = m.forward(x)
                loss = criterion(input=Y_out, target=y)
                loss.backward(retain_graph = True)
                optWhole.zero_grad()

                for param in m.parameters():
                    h_t, i_t, f_t, c_t = trainer.forward(th_t1=param, dL_t = param.grad, L_t = loss,
                                                         h_t1=h_t, i_t1 = i_t, f_t1 = f_t, c_t1=c_t,)
                    # ^ figuire out how to initialize values before first pass for this shit, in the paper
                    param.data = trainer.update().detach()

            x = test_batches[i][:, 1:]
            y = test_batches[i][:, :1]
            Y_out = m.forward(x)
            loss = criterion(input=Y_out, target=y)
            loss.backward(retain_graph=True)
            optTrainer.step()
            optWhole.zero_grad()


def main():
    trainer = ml.MetaLearner()


if __name__ == "__main__":
    main()