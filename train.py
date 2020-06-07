import torch
import os.path
import pandas as pd
import numpy as np
import math
import random
import matplotlib
import matplotlib.pyplot as plt
import torch.nn as nn
from data_helper import labeled_dataset, unlabeled_dataset, image_folder, annotation_csv, labeled_scene_index, transform
from helper import collate_fn
import torch.nn.functional as F
from models import MainURoadMaps
import torch.optim as optimizer
import logging
batch_size = 2
learning_rate = 0.001
trainloader = torch.utils.data.DataLoader(labeled_dataset, batch_size=batch_size, shuffle=True, num_workers=2,
                                          collate_fn=collate_fn)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device is: ', device)
model = MainURoadMaps()
model.to(device)
# print('Model', model.__name__)

optimizer = optimizer.Adam(model.parameters(), lr=learning_rate)

print('entering training loop..')

for epoch in range(10):
    # def train_roadmap_generator(model, optimizer, epoch, log_interval=500,device = device, trainloader = trainloader):
    # Set model to training mode
    model.train()
    # Loop through data points

    for batch_idx, (sample, target, road_image, extra) in enumerate(trainloader):

        model_input = torch.stack(sample)
        # targets = torch.stack(target[batch_idx])

        # convert boolean values tensor to int tensor
        road_maps = tuple(road_image[i].int() for i in range(batch_size - 1))  # batch_size 1

        # [batch_size, 800,800]
        roadmaps = torch.stack(road_maps)
        print('shape of roadmaps is ..', roadmaps.shape)

        # Send data and target to device
        model_input, roadmaps = model_input.to(device), roadmaps.to(device)

        # Zero out the ortimizer
        optimizer.zero_grad()

        # Pass data through model
        preds = model(model_input)
        preds = F.log_softmax(preds, dim=1)

        # Compute binary cross cuz image is binary
        loss = F.binary_cross_entropy(preds, roadmaps)

        # Backpropagate loss
        loss.backward()

        # Make a step with the optimizer
        optimizer.step()

        # save model

        torch.save(model.state_dict(), 'UNet_roadmaps')

        # Print loss
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * 2, len(trainloader.dataset),
                       100. * batch_idx / len(trainloader), loss.item()))

print('Finished Training')
