import torch
import torch.nn as nn

class ShipClassificationCNN(nn.Module):
    """
        Convolutional Neural Network for ship classification.

        Args:
            img_height (int): Height of the input images
            img_width (int): Width of the input images
            num_classes (int): Number of classes in the dataset
            num_layers (int): Number of layers
    """
    def __init__(self, img_height=128, img_width=128, num_classes=10, num_layers=10):
        super(ShipClassificationCNN, self).__init__()
        self.img_height = img_height
        self.img_width = img_width
        self.num_classes = num_classes
        self.num_layers = num_layers

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(128 * (self.img_height // 8) * (self.img_width // 8), 128)
        self.relu_fc = nn.ReLU()
        self.fc2 = nn.Linear(128, self.num_classes)

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

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu_fc(x)
        x = self.fc2(x)

        return x
    
def get_trained_cnn(cnn_path = "best_ship_cnn_model.pth"):
    """
        Returns a trained CNN model.
    """
    model = ShipClassificationCNN()
    model.load_state_dict(torch.load(cnn_path, map_location=torch.device('cpu')))
    return model