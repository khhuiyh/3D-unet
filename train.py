import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
from sklearn import preprocessing
from datetime import datetime, timedelta
import os
import nibabel as nib
from matplotlib import pylab as plt
import cv2
from tensorboardX import SummaryWriter
from skimage.measure import label, regionprops
import csv
from torch.utils.data import Dataset, DataLoader
from loss_fun import IoU, IoULoss_plus_BCEloss
from process_fun import *
from Unet_model import *
import random

n_samples = 10


def printpath(level, path):
    filelistt = []
    files = os.listdir(path)
    for f in files:
        if os.path.isfile(path + '/' + f):
            filelistt.append(f)
    return filelistt


class FracNetTrainDataset(Dataset):

    def __init__(self, image_dir, label_dir=None, crop_size=64,
                 transforms=None, num_samples=4, train=True):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.public_id_list = sorted([x.split("-")[0]
                                      for x in os.listdir(image_dir)])
        self.crop_size = crop_size
        self.transforms = transforms
        self.num_samples = num_samples
        self.train = train

    def __len__(self):
        return len(self.public_id_list)

    @staticmethod
    def _get_pos_centroids(gt):
        gt = label(gt)
        proplist = regionprops(gt)
        poscenter_point = [tuple([round(x) for x in propgt.centroid])
                           for propgt in proplist]

        return poscenter_point

    @staticmethod
    def _get_symmetric_neg_centroids(pos_centroids, x_size):
        symmetric_negcenter = [(x_size - x, y, z) for x, y, z in pos_centroids]

        return symmetric_negcenter

    @staticmethod
    def _get_spine_neg_centroids(shape, crop_size, num_samples):
        x_min, x_max = shape[0] // 2 - 40, shape[0] // 2 + 40
        y_min, y_max = 300, 400
        z_min, z_max = crop_size // 2, shape[2] - crop_size // 2
        spine_neg_centroids = [(
            np.random.randint(x_min, x_max),
            np.random.randint(y_min, y_max),
            np.random.randint(z_min, z_max)
        ) for _ in range(num_samples)]

        return spine_neg_centroids

    def _get_neg_centroids(self, pos_centroids, image_shape):
        num_pos = len(pos_centroids)
        sym_neg_centroids = self._get_symmetric_neg_centroids(
            pos_centroids, image_shape[0])

        if num_pos < self.num_samples // 2:
            spine_neg_centroids = self._get_spine_neg_centroids(image_shape,
                                                                self.crop_size, self.num_samples - 2 * num_pos)
        else:
            spine_neg_centroids = self._get_spine_neg_centroids(image_shape,
                                                                self.crop_size, num_pos)

        return sym_neg_centroids + spine_neg_centroids

    def _get_roi_centroids(self, label_arr):
        pos_centroids = self._get_pos_centroids(label_arr)

        neg_centroids = self._get_neg_centroids(pos_centroids,
                                                label_arr.shape)

        num_pos = len(pos_centroids)
        num_neg = len(neg_centroids)
        global n_samples
        n_samples = self.num_samples
        if num_pos >= self.num_samples:
            num_pos = self.num_samples // 2 + 2
            num_neg = self.num_samples // 2 - 2
        elif num_pos >= self.num_samples // 2:
            num_neg = self.num_samples - num_pos
        # if num_pos >= self.num_samples // 2:
        #     num_pos = self.num_samples // 2
        #     num_neg = self.num_samples // 2

        if num_pos < len(pos_centroids):
            pos_centroids = [pos_centroids[i] for i in np.random.choice(
                range(0, len(pos_centroids)), size=num_pos, replace=False)]
        if num_neg < len(neg_centroids):
            neg_centroids = [neg_centroids[i] for i in np.random.choice(
                range(0, len(neg_centroids)), size=num_neg, replace=False)]

        roi_centroids = pos_centroids + neg_centroids
        random.shuffle(roi_centroids)

        roi_centroids = [tuple([int(x) for x in centroid])
                         for centroid in roi_centroids]

        return roi_centroids

    def _crop_roi(self, arr, centroid):
        flag = 0
        src_beg = [centroid[i] - self.crop_size // 2
                   for i in range(len(centroid)) if centroid[i] - self.crop_size // 2 >= 0]
        src_end = [centroid[i] + self.crop_size // 2
                   for i in range(len(centroid)) if centroid[i] + self.crop_size // 2 <= arr.shape[i]]
        if len(src_beg) < 3 or len(src_end) < 3:
            flag = 1
            return -1, flag
        roi = arr[
              src_beg[0]:src_end[0],
              src_beg[1]:src_end[1],
              src_beg[2]:src_end[2],
              ]

        return roi, flag

    def _apply_transforms(self, image):
        for t in self.transforms:
            image = t(image)

        return image

    def __getitem__(self, idx):
        # read image and label
        public_id = self.public_id_list[idx]
        image_path = os.path.join(self.image_dir, f"{public_id}-image.nii.gz")
        label_path = os.path.join(self.label_dir, f"{public_id}-label.nii.gz")
        image = nib.load(image_path)
        gt_label = nib.load(label_path)
        image_arr = minmax_normalize(image.get_fdata().astype(np.float))
        label_arr = gt_label.get_fdata()
        label_arr[label_arr > 0.5] = 1.0
        # label_arr[label_arr < -0.5] = 1.0

        roi_centroids = self._get_roi_centroids(label_arr)

        image_rois = []
        label_rois = []

        for center in roi_centroids:
            k1, f1 = self._crop_roi(image_arr, center)
            k2, f2 = self._crop_roi(label_arr, center)
            if f1 == 1 or f2 == 1:
                continue
            image_rois.append(k1)
            label_rois.append(k2)
        if len(image_rois) == 0:
            return image_rois, label_rois, -1
        image_rois = torch.tensor(np.stack(image_rois)[:, np.newaxis],
                                  dtype=torch.float)
        label_rois = np.stack(label_rois).astype(np.float)
        label_rois = torch.tensor(label_rois[:, np.newaxis],
                                  dtype=torch.float)

        return image_rois, label_rois, 1


def main():
    end_epoch = 100
    # hardlist = torch.load('hardlist')
    a = 'n'
    schflag = 'y'
    learnning_rate = 0.0001
    dev = 'gpu'
    global n_samples
    n_samples = 12
    batchsize = 12
    mineloss = 0.4372

    filelist_val = printpath(1, "E:/dataset/ribfrac-val/data")[:1]
    file = filelist_val[0]
    label_filename = './ribfrac-val/label/' + file.split('-')[0] + '-label.nii.gz'
    data_filename = './ribfrac-val/data/' + file
    outdir = './prediction/' + file.split('-')[0] + '.nii.gz'
    train_image_dir = "E:/dataset/ribfrac-train/data"
    train_label_dir = "E:/dataset/ribfrac-train/label"
    model = thrdunet(in_channels=1, out_channels=1, num_conv_blocks=2, model_depth=3, dev=dev)
    # model = mythrdunet_transpose()
    criterion = IoULoss_plus_BCEloss()
    iou = IoU()
    bce = nn.BCELoss()

    if a == "y":
        for m in model.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose3d):
                nn.init.xavier_uniform_(m.weight)
        if dev == 'gpu':
            model = model.cuda()
        start_epoch = 0
        n_figures = -1
        epochloss = []
    else:
        # path_checkpoint = r'E:\dataset\checkpoint\checkpoints_loss_0.4444.pth'  # 断点路径
        # path_checkpoint = r'E:\dataset\checkpoint\checkpoints_loss_0.7521.pth'  # 断点路径
        path_checkpoint = './checkpoint/checkpoints_new.pth'
        checkpoint = torch.load(path_checkpoint)  # 加载断点
        epochloss = checkpoint['epochloss']

        model.load_state_dict(checkpoint['net'])  # 加载模型可学习参数

        start_epoch = checkpoint['epoch']  # 设置开始的epoch
        n_figures = checkpoint['num_figures']
    if dev == 'gpu':
        model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=learnning_rate)
    if schflag == 'y':
        # schedeler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 6, 0.0001)
        schedeler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5,
                                                               patience=8, verbose=True, threshold=0.001, min_lr=0.000001)
    # ds_train = FracNetTrainDataset(train_image_dir, train_label_dir, num_samples=n_samples)
    # ds_train = torch.load('mytensor')
    for epoch in range(0, end_epoch):
        print("| learningrate", format(optimizer.param_groups[0]['lr'], '.6f'))
        for idx in range(0, 420):
        # for idx in hardlist:
            # x_tra, y_tra, flag = ds_train[idx]
            # if flag == -1:
            #     continue
            x_tra, y_tra = torch.load('E:/dataset/data/mytensor'+str(idx))
            # if epoch == 0:
            #     torch.save((x_tra, y_tra), 'E:/dataset/data/mytensor'+str(idx))
            # for i in range(tuple(x_tra.shape)[0]):
            #     x_train = torch.unsqueeze(x_tra[i], dim=0)
            #     y_train = torch.unsqueeze(y_tra[i], dim=0)
            #     x_train, y_train = data_processing(x_train, y_train)
            for slice_start in range(0, x_tra.shape[0], batchsize):
                slice_stop = min(slice_start + batchsize, x_tra.shape[0])
                x_train, y_train = x_tra[slice_start:slice_stop].cuda(), y_tra[slice_start:slice_stop].cuda()

                optimizer.zero_grad()
                torch.cuda.empty_cache()
                y_pred = model(x_train)
                # y_pred = torch.ones(4,1,64,64,64)*0.005
                # y_train = torch.zeros(4,1,64,64,64)*0.0
                loss = criterion.forward(inputs=y_pred, targets=y_train)
                epochloss.append(loss.cpu().item())

                # bbb = bce(y_pred, y_train).cpu().item()
                # iou_metrics = iou.forward(inputs=(y_pred > 0.5).float(), targets=y_train)
                # print('| epoch', epoch,
                #       '| num_figures', idx,
                #       '| num_samples', n_samples,
                #       '| input size', x_train.shape,
                #       '| BCEloss ', bbb,
                #       '| loss', loss.cpu().item(),
                #       '| iou', iou_metrics.cpu())

                torch.cuda.empty_cache()
                loss.backward()
                optimizer.step()
            if idx % 100 == 0 and idx > 0:
                print("saving model...")
                checkpoint = {
                    "net": model.state_dict(),
                    "epoch": epoch,
                    'num_figures': idx,
                    'epochloss': epochloss,
                    'learninglate': optimizer.param_groups[0]['lr']
                }
                if not os.path.isdir("./checkpoint"):
                    os.mkdir("./checkpoint")
                torch.save(checkpoint, './checkpoint/checkpoints_new.pth')
        lll = len(epochloss)
        epochloss = np.average(np.array(epochloss))
        print("epoch loss:", epochloss, lll)
        if schflag == 'y':
            schedeler.step(epochloss)
        if epochloss < mineloss:
            mineloss = epochloss
            print("saving model...", "epoch loss:", mineloss)
            checkpoint = {
                "net": model.state_dict(),
                "epoch": epoch + 1,
                'num_figures': -1,
                'learninglate': optimizer.param_groups[0]['lr'],
                'epochloss': []
            }
            if not os.path.isdir("./checkpoint"):
                os.mkdir("./checkpoint")
            torch.save(checkpoint, './checkpoint/checkpoints_loss_' + str(format(mineloss, '.4f')) + '.pth')
        epochloss = []
        n_figures = -1


if __name__ == "__main__":
    main()
