import Model.MetaLearner as ml
import Model.Segmenter as seg
import torch
import math

def getDataSets():
    pass

def trainTrainer(batches = 10):
    """
    :param batches: if data can't fit on gpu then increase number of batches, implemented so that whole batch is ran on gpu

    :return:
    """
    trainer = ml.MetaLearner()

    data = getDataSets()

    for d in data:
        model = seg.Segmenter(d.channels)
        opt = torch.optim.Adam(model.parameters(), lr=.00005, weight_decay=.005)
        criterion = torch.nn.BCELoss()

        x_full = d.getX()
        n_train = x.shape[0]
        y_full = d.getY()

        n = int(math.ceil(len(d)/batches))
        batches = [d[i:min(i+n, len(d))] for i in range(0, len(d), n)]
        train_batches = [b[:-len(b)//2] for b in batches]
        test_batches = [b[len(b)//2:] for b in batches]

        # potentially k fold on each batch in the future

        for i in range(0, len(train_batches)):
            x = train_batches[i][:,1:]
            y = train_batches[i][:,:1]

            opt.zero_grad() #fix, do

            Y_out = model.forward(x)
            Y_out_d = Y_out.detach() # if test then don't detach we want the metalearner to update its parameters
            loss = criterion(input=Y_out, target=y)
            loss.backward(retain_graph = False)

            for param in model.parameters():
                h_t, i_t, f_t, c_t = trainer.forward(th_t1=param, dL_t = param.grad, L_t = loss,
                                                     h_t1=h_t, i_t1 = i_t, f_t1 = f_t, c_t1=c_t,)
                # ^ figuire out how to initialize values before first pass for this shit, in the paper

                param.data = trainer.update()

            x_test = test_batches[i][:,1:]
            y_test = test_batches[i][:,1:]

            Y_out = model.forward(x)
            Y_out_d = Y_out.detach() # if test then don't detach we want the metalearner to update its parameters
            loss = criterion(input=Y_out_d, target=y)
            loss.backward(retain_graph = False)
            dL = loss.grad



def main():
    # print(5/6)
    # print(11//6)
    trainer = ml.MetaLearner()


if __name__ == "__main__":
    main()