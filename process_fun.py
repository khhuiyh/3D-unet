import os
import torch
import numpy as np
import random


def printpath(level, path):
    filelistt = []
    files = os.listdir(path)
    for f in files:
        if os.path.isfile(path + '/' + f):
            filelistt.append(f)
    return filelistt


def minmax_normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def data_processing(data, label):
    data, label = np.array(torch.squeeze(data), dtype=float), np.array(torch.squeeze(label), dtype=float)
    datalist = [data]
    labellist = [label]
    noise = np.random.randn(data.shape[0], data.shape[1], data.shape[2]) * 0.1
    datalist.append(minmax_normalize(data + noise))
    labellist.append(label)

    k_flip = np.random.randint(0, 3)
    datalist.append(np.flip(data, axis=k_flip))
    labellist.append(np.flip(label, axis=k_flip))

    k_rot = np.random.randint(0, 4)
    datalist.append(np.rot90(data, k_rot))
    labellist.append(np.rot90(label, k_rot))

    data = torch.tensor(np.stack(datalist)[:, np.newaxis], dtype=torch.float)
    label = torch.tensor(np.stack(labellist)[:, np.newaxis], dtype=torch.float)

    return data, label
