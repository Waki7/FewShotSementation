from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import itertools
import seaborn as sns
import Model.Config as cfg


def plot_confusion_matrix(predictions, ground_truths,
                          results_qualifier,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure()
    cm = calc_confusion_matrix(predictions, ground_truths)
    classes = [str(i) for i in range(0, cm.shape[0])]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    ax = sns.heatmap(cm, linewidth=0.5, cmap="YlGnBu")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(cfg.graph_file_path + results_qualifier + '.png', format='png', dpi=1000)


def calc_confusion_matrix(predictions, ground_truths):
    predictions = predictions.flatten()
    ground_truths = ground_truths.flatten()
    return confusion_matrix(ground_truths, predictions)

def accuracy(predictions, ground_truths, ignore_index = None):
    accuracies = []
    for pred, truth in zip(predictions, ground_truths):
        accuracy = np.sum(pred == truth)
        if ignore_index is not None:
            total = np.sum(truth != ignore_index)
        else:
            total = np.product(truth.shape)
        accuracies.append(accuracy/total)
    return np.average(accuracies)