import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from cnn_model import get_trained_cnn
from torch.autograd import Variable
from utils import get_datasets

def get_active_neurons(conv_layer, activation_layer, input_data):
    with torch.no_grad():
        output = activation_layer(conv_layer(input_data))
        print(f"Layer {conv_layer}: Output shape = {output.shape}")
        active_neurons = output.abs().sum(dim=[0, 2, 3]) > 0
    return active_neurons, output

class PrunedCNN(nn.Module):
    def __init__(self, original_model, target_class, img_height=128, img_width=128):
        super(PrunedCNN, self).__init__()

        active_neurons_conv1, output1 = get_active_neurons(original_model.conv1, original_model.relu1, sample_input)
        output1 = original_model.pool1(output1)
        active_neurons_conv2, output2 = get_active_neurons(original_model.conv2, original_model.relu2, output1)
        output2 = original_model.pool2(output2)
        active_neurons_conv3, output3 = get_active_neurons(original_model.conv3, original_model.relu3, output2)
        output3 = original_model.pool3(output3)

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=active_neurons_conv1.sum().item(),
                               kernel_size=original_model.conv1.kernel_size, stride=original_model.conv1.stride,
                               padding=original_model.conv1.padding)
        self.conv1.weight.data = original_model.conv1.weight[active_neurons_conv1]
        self.conv1.bias.data = original_model.conv1.bias[active_neurons_conv1]
        self.relu1 = original_model.relu1
        self.pool1 = original_model.pool1

        self.conv2 = nn.Conv2d(in_channels=self.conv1.out_channels, out_channels=active_neurons_conv2.sum().item(),
                               kernel_size=original_model.conv2.kernel_size, stride=original_model.conv2.stride,
                               padding=original_model.conv2.padding)
        self.conv2.weight.data = original_model.conv2.weight[active_neurons_conv2]
        self.conv2.bias.data = original_model.conv2.bias[active_neurons_conv2]
        self.relu2 = original_model.relu2
        self.pool2 = original_model.pool2

        self.conv3 = nn.Conv2d(in_channels=self.conv2.out_channels, out_channels=active_neurons_conv3.sum().item(),
                               kernel_size=original_model.conv3.kernel_size, stride=original_model.conv3.stride,
                               padding=original_model.conv3.padding)
        self.conv3.weight.data = original_model.conv3.weight[active_neurons_conv3][:, active_neurons_conv2]
        self.conv3.bias.data = original_model.conv3.bias[active_neurons_conv3]
        self.relu3 = original_model.relu3
        self.pool3 = original_model.pool3

        #Dummy example to know what size fc1 needs to be
        dummy_input = torch.zeros(sample_input.shape)
        with torch.no_grad():
            dummy_out = self.pool3(self.relu3(self.conv3(
                self.pool2(self.relu2(self.conv2(
                    self.pool1(self.relu1(self.conv1(dummy_input)))))))))
        flattened_size = dummy_out.view(1, -1).shape[1]

        self.fc1 = nn.Linear(flattened_size, 128)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = x.view(1, -1)
        x = self.fc1(x)
        x = self.fc2(x)

        return self.sigmoid(x)

if __name__ == "__main__":
    target_class = 6
    dataset_example_number = 6060
    original_model = get_trained_cnn()
    dataset, _, _ = get_datasets()
    print("Dataset image example class : " + str(dataset[dataset_example_number][1]))
    sample_input = dataset[dataset_example_number][0].unsqueeze(0)
    pruned_model = PrunedCNN(original_model, target_class, sample_input)
    print(pruned_model)

    output = pruned_model(sample_input)
    print("Output:", output)