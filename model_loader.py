"""
You need to implement all four functions in this file and also put your team info as a variable
Then you should submit the python file with your model class, the state_dict, and this file
/Users/kunal2/PycharmProjects/Deep_Learning_Project/code/model_loader.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from models import *
# import your model class
import torch
import torchvision.models as models


# Put your transform function here, we will use it for our dataloader
def get_transform():
    return torchvision.transforms.Compose([transforms.RandomCrop((224, 224)),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])])



class ModelLoader(nn.Module):
    # Fill the information for your team
    team_name = 'Team 51'
    team_member = ['tg1779', 'kag634']
    contact_email = '@nyu.edu'

    def __init__(self,filename = 'models'):
        super(ModelLoader, self).__init__()

        #
        #         #       1. create the model object
        #         #       2. load your state_dict
        #         #       3. call cuda()
        #         # self.model = ...
        #         #
        cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if cuda else "cpu")

        # load UNet for roadmap predictions
        self.roadmap_model = MainURoadMaps()
        self.roadmap_model.load_state_dict(torch.load(filename)['Net_roadmap'])  # roadmaps
        self.roadmap_model.to(self.device)
        self.category_model = MainUCategories()
        self.category_model.load_state_dict(torch.load(filename)['Net_category'])


    def get_bounding_boxes(samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a tuple with size 'batch_size' and each element is a cuda tensor [N, 2, 4]
        # where N is the number of object
        return torch.rand(1, 15, 2, 4) * 10
        pass

    def get_binary_road_map(self, samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a cuda tensor with size [batch_size, 800, 800] 
        model_input = torch.stack(samples)

        # Send data to device
        model_input = model_input.to(self.device)

        # Pass data through model
        output = self.roadmap_model(model_input)
        output = F.log_softmax(output, dim=1)

        # Get predictions (indices) from the model for each data point
        model_preds = output.data.max(1, keepdim=True)[1]

        return model_preds.squeeze(1)  # [batch_size, 800, 800]
        pass


