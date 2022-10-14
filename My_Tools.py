import numpy as np
import math
import pickle
from tqdm import *
from My_Opt import opt
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import torch
import os
# from scipy.signal import savgol_filter

# ---------------------------------------------------------

def open_data_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    return data

# ---------------------------------------------------------

class ConfusionMatrix(object):
    """
    注意，如果显示的图像不全，是matplotlib版本问题
    本例程使用matplotlib-3.2.1(windows and ubuntu)绘制正常
    需要额外安装prettytable库
    """
    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def summary(self):
        # calculate accuracy
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = round(sum_TP / np.sum(self.matrix), 5)
        print("the model accuracy is ", acc)

        # precision, recall, specificity
        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "Specificity"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP

            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 5) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 5) if TP + FN != 0 else 0.
            Specificity = round(TN / (TN + FP), 5) if TN + FP != 0 else 0.
            table.add_row([self.labels[i], Precision, Recall, Specificity])
        print(table)

    def plot(self):
        plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
        plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内
        matrix = self.matrix
        print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)


        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels)

        # 显示colorbar
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('TMI loss Confusion matrix r=10s')

        # 在图中标注数量/概率信息
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.show()


def evalute(model, loader):
    correct = 0
    total = len(loader.dataset)

    for seg_cut_in_8, seg_cut_in_6, seg_cut_in_4, global_data, label in tqdm(loader):
        # if opt['device'] == 'cuda':
        #     data, label = data.cuda(), label.cuda()
        with torch.no_grad():
            logits = model(seg_cut_in_8, seg_cut_in_6, seg_cut_in_4, global_data)
            pred = logits.argmax(dim=1)
            true_label = label.argmax(dim=1)
        correct += torch.eq(pred, true_label).sum().float().item()

    return correct / total


def evalute_with_graph(model, loader):
    correct = 0
    total = len(loader.dataset)
    label_list = ['walk and run', 'bike', 'bus', 'car or taxi', 'railway(include subway and train)']
    # label_list = ['步行', '自行车', '开车', '火车（包括地铁）', '摩托', '其他（包括轮船和飞机）']
    confusion = ConfusionMatrix(num_classes=5, labels=label_list)
    for seg_cut_in_8, seg_cut_in_6, seg_cut_in_4, global_data, label in tqdm(loader):
        # if opt['device'] == 'cuda':
        #     data, label = data.cuda(), label.cuda()
        with torch.no_grad():
            logits = model(seg_cut_in_8, seg_cut_in_6, seg_cut_in_4, global_data)
            pred = logits.argmax(dim=1)
            true_label = label.argmax(dim=1)
            confusion.update(pred.to('cpu').numpy(), true_label.to('cpu').numpy())
        correct += torch.eq(pred, true_label).sum().float().item()
    confusion.plot()
    confusion.summary()

    return correct / total


def adjust_learning_rate(optimizer, epoch, start_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = start_lr * (0.1 ** (epoch // 3))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# ------------------My_Model用的----------------------------


def choice_weight(weight_dir_path):
    weight_file_list = os.listdir(weight_dir_path)
    weight_num = 0
    best_acc_name = 0

    for i, name in enumerate(weight_file_list):
        tmp = name.split('.m')
        file_name_list = tmp[0].split(',')
        if len(file_name_list) == 3:
            acc_now = float(file_name_list[1].split('%')[0])
            if acc_now >= best_acc_name:
                weight_num = i
                best_acc_name = acc_now
    weight_path = os.path.join(weight_dir_path, weight_file_list[weight_num])

    return weight_path
