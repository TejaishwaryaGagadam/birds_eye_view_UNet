# coding: utf-8

# In[ ]:
import os.path
import pandas as pd
import numpy as np
import math
# import scipy.ndimage as ndimage
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset
from data_helper import UnlabeledDataset, LabeledDataset, transform
from data_helper import labeled_scene_index, unlabeled_scene_index, image_folder,annotation_csv
from helper import collate_fn, draw_box


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv3d) or isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


class DownConvNN(nn.Module):  # ConvNN_1
    def __init__(self, input_channels, output_channels ): # [N,Cin=6,D,H,W]
        if input_channels is None:
            input_channels = 6
        self.input_channels = input_channels
        super(DownConvNN, self).__init__()
        self.bn1 = nn.BatchNorm3d(input_channels)
        self.conv1 = nn.Conv3d(input_channels, 6 * input_channels, kernel_size=3) # depthwise multiplier convolution Cout =  k* Cin
        self.relu1 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm3d(6*input_channels)
        self.conv2 = nn.Conv3d(6 * input_channels, 36 * input_channels, kernel_size = 3)
        self.relu2 = nn.ReLU(inplace=True)
        self.bn3 = nn.BatchNorm3d(36 * input_channels)
        self.conv3 = nn.Conv3d(36 * input_channels, output_channels, kernel_size=3)
        self.relu3 = nn.ReLU()
        self.max1 = nn.MaxPool3d(kernel_size=4, stride=2, padding=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.bn2(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.bn3(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.max1(x)
        return x


class UpConvNN(nn.Module):  # UpConvNN_1
    def __init__(self, input_channels, intermediate_channels, output_channels):
        super(UpConvNN, self).__init__()

        self.bn1 = nn.BatchNorm3d(input_channels)
        self.conv1 = nn.Conv3d(input_channels, 6 * intermediate_channels, kernel_size=3)
        self.relu1 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm3d(6 * intermediate_channels)
        self.conv2 = nn.Conv3d(6 * intermediate_channels, 6 * intermediate_channels, kernel_size=3)
        self.relu2 = nn.ReLU(inplace=True)
        self.bn3 = nn.BatchNorm3d(6 * intermediate_channels)
        self.upconv = nn.ConvTranspose3d(intermediate_channels, output_channels, kernel_size=2),  # stride=2)
        self.relu3 = nn.ReLU(inplace=True)


    def forward(self, x):
        x = self.bn1(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.bn2(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.bn3(x)
        x = self.upconv(x)
        x = self.relu3(x)
        return x



class Last(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Last, self).__init__()
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, 1, kernel_size=3) # [N,Cin,D,H,W] --> [N,1,D,H,W]
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(1, out_channels, kernel_size=3) # [N,1,D,H,W] -- > [N,1,Dout,H,W]
        self.rel3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.bn1(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.bn2(x)
        x = self.conv2(x)
        x = self.relu2(x)
        return x


class Final(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(Final, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels, n_classes, 1)


    def forward(self, x):
        x = self.conv1(x)
        return x


class MainURoadMaps(nn.Module):
    def __init__(self,bilinear = True):
        super(MainURoadMaps, self).__init__()

        self.down1 = DownConvNN(6, 16)
        self.down2 = DownConvNN(32, 64)
        self.down3 = DownConvNN(64, 128)
        # self.down4 = DownConvNN(128, 256)
        # self.down5 = DownConvNN(256, 256)
        # self.up1 = UpConvNN(256, 512, 256)
        self.up1 = UpConvNN(128, 256, 128)
        self.up2 = UpConvNN(256, 128, 64)
        self.up3 = UpConvNN(128, 64, 32)
        self.last = Last(32, 1) # flattening the feature space from 5d to 4d
        initialize_weights(self)

    def forward(self, x):
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        # down4 = self.down4(down3)
        # down5 = self.down5(down4)
        # up1 = self.up1(down5, down4)
        up1 = self.up1(down3, down2)
        up2 = self.up2(up1, down1)
        unet_out = self.last(up2) # [N,Cout = 1, D = 1,H,W]
        return F.interpolate(unet_out, 800, mode='bilinear') # upsample to the dimension of road map

# class MainUCategories(nn.Module):
#     def __init__(self, bilinear=True):
#         super(MainUCategories, self).__init__()
#
#         self.down1 = ConvNN1(n_classes, 64, 128)
#         self.down2 = ConvNN1(64, 128, 256)
#         self.down3 = ConvNN1(128, 256, 512)
#         self.down4 = ConvNN1(256, 512, 1024)
#         self.down5 = ConvNN1(512, 512, 1024)
#         self.up1 = UpConvNN1(512, 1024, 512, bilinear)
#         self.up2 = UpConvNN1(1024, 512, 256, bilinear)
#         self.up3 = UpConvNN1(512, 256, 128, bilinear)
#         self.up4 = UpConvNN1(256, 128, 64, bilinear)
#         self.last = Last(128, 64)
#         self.final = nn.Linear(64, 9)
#         initialize_weights(self)
#
#
# def forward(self, x):
#     down1 = self.down1(x)
#     down2 = self.down2(down1)
#     down3 = self.down3(down2)
#     down4 = self.down4(down3)
#     down5 = self.down5(down4)
#     up1 = self.up1(down5, down4)
#     up2 = self.up2(up1, down3)
#     up3 = self.up3(up2, down2)
#     up4 = self.last(up3, down1)
#     final = self.final(up4)
#     return final
#

def bbox_to_label(target_object):
    categories = target_object[0]['category']
    bboxes = target_object[0]['bounding_box']

    output = np.zeros((800, 800))
    # print(len(categories))

    for i in range(len(bboxes)):
        class_label = categories[i]
        bounding_box = bboxes[i]
        flx, frx, blx, brx = bounding_box[0]
        fly, fry, bly, bry = bounding_box[1]
        fx = math.floor(10 * ((flx + frx) / 2) + 400)
        bx = math.floor(10 * ((blx + brx) / 2) + 400)
        fy = math.floor(10 * ((fly + bly) / 2) + 400)
        by = math.floor(10 * ((fry + bry) / 2) + 400)
        output[fy:by, fx:bx] = class_label
        output[by:fy, bx:fx] = class_label
    return output

def generate_roadmap_image_labels(target_object):
    categories = target_object[0]['category']
    bboxes = target_object[0]['bounding_box']

    output = np.zeros((800, 800))

    for i in range(len(bboxes)):
        label = categories[i]
        this_bbox = bboxes[i]
        flx, frx, blx, brx = this_bbox[0]
        fly, fry, bly, bry = this_bbox[1]

        # convert the co-ordinates because the above co-ordinates are referenced from the ego car....

        fx = math.floor(10 * ((flx + frx) / 2))
        fx = fx + 400
        bx = math.floor(10 * ((blx + brx) / 2))
        bx = bx + 400
        fy = math.floor(10 * ((fly + bly) / 2))
        fy = fy + 400
        by = math.floor(10 * ((fry + bry) / 2))
        by = by + 400

        output[fy:by, fx:bx] = label # road_map image labels.
        output[by:fy, bx:fx] = label

    return output

def return_images(image_object):
    img = image_object[0]
    front = torch.cat((img[0], img[1], img[2]), 2)
    back = torch.cat((img[5], img[4], img[3]), 2)
    all_images = torch.cat((front, back), 1)
    all_images = all_images.unsqueeze(0)
    return all_images


class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_model(model):
    for param in model.parameters():
        param.requires_grad = True


