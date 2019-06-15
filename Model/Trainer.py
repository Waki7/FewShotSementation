import Model.MetaLearner as ml
import Model.Segmenter as seg
import torch

def getDataSets():
    pass


def train():
    trainer = ml.MetaLearner()

    data = getDataSets()

    batch_size = 10000
    for d in data:
        model = seg.Segmenter(d.channels)
        optimizer = torch.optim.Adam(model.parameters(), lr=.00005, weight_decay=.005)
        criterion = torch.nn.BCELoss()

        x_full = d.getX()
        n_train = x.shape[0]
        y_full = d.getY()
        for i in range(0, n_train, batch_size):
            x = x_full[i:i + batch_size] # assuming it cuts off correctly, we can check that later
            y = y_full[i:i+batch_size]

            Y_out = model.forward(x)
            Y_out_d = Y_out.detach() # if test then don't detach we want the metalearner to update its parameters
            loss = criterion(input=Y_out_d, target=y)
            loss.backward()
            dL = loss.grad

            for param in model.parameters():
                h_t, i_t, f_t, c_t = trainer.forward(dL, h_t1, loss, param, i_t1, f_t1, c_t1)  # figuire out how to initialize values before first pass for this shit, in the paper

                param.data = c_t




def main():
    trainer = ml.MetaLearner()


if __name__ == "__main__":
    main()