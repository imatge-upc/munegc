"""
    2D–3D Geometric Fusion network using Multi-Neighbourhood Graph Convolution for RGB-D indoor scene classification
    2021 Albert Mosella-Montoro <albert.mosella@upc.edu>
"""

import os
import re
from mpl_toolkits.axes_grid1 import make_axes_locatable
import textwrap
import json
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools
import matplotlib.pyplot as plt
plt.switch_backend('agg')


class AverageMeter():
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, count=1):
        self.val = val
        self.sum += val*count
        self.count += count
        self.avg = self.sum / self.count


class ConfusionMatrixMeter():
    def __init__(self, labels, cmap='orange'):
        self._cmap = cmap
        self._k = len(labels)
        self._labels = labels
        self._cm = np.ndarray((self._k, self._k), dtype=np.int32)
        self.reset()

    def reset(self):
        self._cm.fill(0)

    def add(self, target, predicted):

        assert predicted.shape[0] == target.shape[0], \
            'number of targets and predicted outputs do not match'
        if np.ndim(predicted) != 1:
            assert predicted.shape[1] == self._k, \
                'number of predictions does not match size of confusion matrix'
            predicted = np.argmax(predicted, 1)
        else:
            assert (predicted.max() < self._k) and (predicted.min() >= 0), \
                'predicted values are not between 1 and k'
        self._cm += confusion_matrix(target, predicted, labels=range(0, self._k))

    def value(self, normalize=False):
        if normalize:
            np.set_printoptions(precision=2)
            return np.divide(self._cm.astype('float'), self._cm.sum(axis=1).clip(min=1e-12)[:, np.newaxis])
        else:
            return self._cm

    def accuracy(self):
        return np.divide(self.value().trace(), self.value().sum())*100

    def mean_acc(self):
        return np.divide(self.value(True).trace(), self._k)*100

    def save_json(self, filename, normalize=False):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w', encoding="utf8") as f:
            json.dump(self.value(normalize=normalize).tolist(), f)

    def save_npy(self, filename, normalize=False):
        np.save(filename, self.value())

    def plot(self, normalize=False):
        cm = self.value(normalize=normalize)
        fig = plt.figure(figsize=(self._k, self._k), dpi=100, facecolor='w', edgecolor='k')
        ax = fig.add_subplot(1, 1, 1)
        if normalize:
            cm_plot = ax.imshow(cm, cmap=self._cmap, vmin=0, vmax=1)
        else:
            cm_plot = ax.imshow(cm, cmap=self._cmap)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        ax.figure.colorbar(cm_plot, cax=cax)
        classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x) for x in self._labels]
        classes = ['\n'.join(textwrap.wrap(l, 20)) for l in classes]
        tick_marks = np.arange(len(classes))

        ax.set_xlabel('Predicted')
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(classes, rotation=-90, ha='center')
        ax.xaxis.set_label_position('bottom')
        ax.xaxis.tick_bottom()

        ax.set_ylabel('True Label')
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(classes, va='center')
        ax.yaxis.set_label_position('left')
        ax.yaxis.tick_left()

        for i, j in itertools.product(range(self._k), range(self._k)):
            if normalize:
                ax.text(j, i, np.round(cm[i, j], 2), horizontalalignment="center",
                        verticalalignment='center', color="black")
            else:
                ax.text(j, i, int(cm[i, j]) if cm[i, j] != 0 else '.',
                        horizontalalignment="center", verticalalignment='center', color="black")
        fig.set_tight_layout(True)
        return fig
