import torch
import torch.nn as nn
from utils import get_datasets

class ShipClassificationCNN(nn.Module):
    """
        Convolutional Neural Network for ship classification.

        Args:
            img_height (int): Height of the input images
            img_width (int): Width of the input images
            num_classes (int): Number of classes in the dataset
            num_layers (int): Number of layers
    """
    def __init__(self, img_height=128, img_width=128, num_classes=10, num_layers=6):
        super(ShipClassificationCNN, self).__init__()
        self.img_height = img_height
        self.img_width = img_width
        self.num_classes = num_classes
        self.num_layers = num_layers

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * (self.img_height // 8) * (self.img_width // 8), 128)
        self.relu_fc = nn.ReLU()
        self.fc2 = nn.Linear(128, self.num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        size1 = x.size()
        x, indices1 = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        size2 = x.size()
        x, indices2 = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        size3 = x.size()
        x, indices3 = self.pool3(x)

        # Save pool output sizes and indices for later inversion
        self.pool_info = {
            'pool1': {'indices': indices1, 'output_size': size1},
            'pool2': {'indices': indices2, 'output_size': size2},
            'pool3': {'indices': indices3, 'output_size': size3}
        }

        x = self.flatten(x)
        x = x.view(1, -1)
        x = self.fc1(x)
        x = self.relu_fc(x)
        x = self.fc2(x)

        return x
    
def get_trained_cnn(cnn_path = "best_ship_cnn_model.pth"):
    """
        Returns a trained CNN model.
    """
    model = ShipClassificationCNN()
    model.load_state_dict(torch.load(cnn_path, map_location=torch.device('cpu'), weights_only=True))
    return model
if __name__ == "__main__":
    model = get_trained_cnn()
    # model = ShipClassificationCNN()
    # weights = torch.load(
    #      "best_ship_cnn_model.pth", map_location=torch.device('cpu'), weights_only=True
    #  )
    # model.load_state_dict(weights)
    dataset, _, _ = get_datasets()
    inp = dataset[0]
    print(model)
    print(inp[0].shape)
    print(model(inp[0]))
