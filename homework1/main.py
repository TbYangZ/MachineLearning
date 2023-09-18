import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.metrics import roc_curve, auc

path = r'.\MachineLearning\homework1'

filename = path + r'\result.txt'
with open(filename, 'r') as file:
    lines = file.readlines()

class Data:
    def __init__(self, number, state):
        self.number = number
        self.state = state
    def __lt__(self, other):
        return self.number > other.number

arr = []
for line in lines:
    number = line.strip().split()
    rate = float(number[0])
    if number[1] == 'True':
        y = 1
    else:
        y = 0
    arr.append(Data(rate, y))

arr = np.array(arr)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


arr = np.array([Data(sigmoid(x.number), x.state) for x in arr])

arr = sorted(arr)

y_true = [x.state for x in arr]
y_score = [x.number for x in arr]

def DrawUseLib():
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig(path + r'\res_2.png')
    plt.show()

def DrawUseDefinition():
    nowx, nowy, tot = 0, 0, 0
    for i in range(len(arr)):
        if y_true[i] == 1:
            tot += 1

    x = []
    y = []

    for i in range(len(arr)):
        if y_true[i] == 1:
            nowy += 1.0 / tot
        else:
            nowx += 1.0 / (len(arr) - tot)
        x.append(nowx)
        y.append(nowy)

    plt.figure()
    plt.plot(x, y, color='darkorange', label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig(path + r"\res_1.png")
    plt.show()