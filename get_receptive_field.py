import torch
import torch.nn as nn
import numpy as np
from cnn_model import get_trained_cnn
from utils import get_datasets

class ReducedCNN(nn.Module):
    def __init__(self, original_model, target_class, img_height=128, img_width = 128):
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

        #self.fc1 = original_model.fc1
        self.fc1 = nn.Linear(96 * (img_height // 8) * (img_width // 8), 96)

        self.sigmoid = nn.Sigmoid()
        print(f"Original fc1 weight shape: {original_model.fc1.weight.shape}")
        print(f"Original fc2 weight shape: {original_model.fc2.weight.shape}")
        #Weights for the target class
        fc2_weights = original_model.fc2.weight[target_class, :]
        threshold = torch.quantile(fc2_weights.abs(), 0.25) #threshold for selecting important neurons for the moment = 75% kept
        important_neurons = fc2_weights.abs() > threshold
        print("important neurons", important_neurons)

        print(f"Number of important neurons selected: {important_neurons.sum().item()}")

        #Reduce fc1 weights by selecting only important neurons
        reduced_fc1_weights = original_model.fc1.weight[:, important_neurons]
        print(f"Original fc1 weight shape: {original_model.fc1.weight.shape}")
        print(f"Reduced fc1 weight shape: {reduced_fc1_weights.shape}")
        
        #Update using reduced weights
        #self.fc1 = nn.Linear(reduced_fc1_weights.shape[1], reduced_fc1_weights.shape[0])
        self.fc1.weight.data = reduced_fc1_weights

        #Reduce conv3 weights by selecting only important neurons
        reduced_conv3_weights = original_model.conv3.weight[important_neurons, :, :, :]
        self.conv3.weight.data = reduced_conv3_weights
        self.conv3.bias.data = original_model.conv3.bias[important_neurons] #keep the original biases (128)

        #self.fc2 = nn.Linear(reduced_fc1_weights.shape[0], 1) # We only need to output a single value, which is a predictor of the probability of the class X
        self.fc2 = nn.Linear(128, 1)

        target_class_weights = fc2_weights[target_class].unsqueeze(0)
        self.fc2.weight.data = target_class_weights.view(1, -1)
        self.fc2.bias.data = torch.tensor([original_model.fc2.bias[target_class].item()])

    def forward(self, x):
        x = self.conv1(x)
        print(f"Shape after conv1: {x.shape}")
        x = self.relu1(x)
        print(f"Shape after relu1: {x.shape}")
        x = self.pool1(x)
        print(f"Shape after pool1: {x.shape}")
        
        x = self.conv2(x)
        print(f"Shape after conv2: {x.shape}")
        x = self.relu2(x)
        print(f"Shape after relu2: {x.shape}")
        x = self.pool2(x)
        print(f"Shape after pool2: {x.shape}")

        x = self.conv3(x)
        print(f"Shape after conv3: {x.shape}")
        x = self.relu3(x)
        print(f"Shape after relu3: {x.shape}")  
        x = self.pool3(x)


        print(f"Shape before flatten: {x.shape}")

        x = x.view(1, -1) #flatten
        print(f"Shape after flatten: {x.shape}")
        print("fc1", self.fc1.weight.shape)
        x = self.fc1(x)
        print(f"Shape after fc1: {x.shape}")
        x = self.fc2(x)
        print(f"Shape after fc2: {x.shape}")

        x = self.sigmoid(x)#to transform the final output to a probability of the image representing class X

        return x

if __name__ == "__main__":
    target_class = 3

    original_model = get_trained_cnn()
    reduced_model = ReducedCNN(original_model, target_class)

    print(f"New model for class {target_class} created.")
    print(reduced_model)

    dataset, _, _ = get_datasets() #from utils.py
    inp = dataset[0][0]
    output = reduced_model(inp)
    
    # Print the output shape (should be a single value for the target class)
    print(f"Output shape: {output.shape}")