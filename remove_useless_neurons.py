import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from cnn_model import get_trained_cnn
from torch.autograd import Variable

class ReducedCNN(nn.Module):
    def __init__(self, original_model, target_class, img_height=128, img_width=128):
        super(ReducedCNN, self).__init__()

        self.conv1 = original_model.conv1
        self.relu1 = original_model.relu1
        self.pool1 = original_model.pool1

        self.conv2 = original_model.conv2
        self.relu2 = original_model.relu2
        self.pool2 = original_model.pool2

        self.conv3 = original_model.conv3
        self.relu3 = original_model.relu3
        self.pool3 = original_model.pool3

        self.fc1 = original_model.fc1
        self.fc2 = original_model.fc2

        self.target_class = target_class

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = x.view(-1, 128)
        x = self.fc1(x)
        x = self.fc2(x)

        return self.sigmoid(x)

        def get_saliency_map(self, input_image):
            self.eval()

            input_image