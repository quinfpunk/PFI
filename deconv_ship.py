import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from tqdm import tqdm
from cnn_model import get_trained_cnn
from utils import get_datasets

class DeconvNet(nn.Module):
    def __init__(self, cnn):
        super(DeconvNet, self).__init__()

        self.unpool3 = nn.MaxUnpool2d(2, stride=2)
        # Deconvolution: Note that input channels become output channels and vice-versa.
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        # Mirror the second block first
        self.unpool2 = nn.MaxUnpool2d(2, stride=2)
        # Deconvolution: Note that input channels become output channels and vice-versa.
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        
        # Mirror the first block
        self.unpool1 = nn.MaxUnpool2d(2, stride=2)
        self.deconv1 = nn.ConvTranspose2d(32, 3, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        
        # Tie deconv weights to CNN’s conv filters (transpose them)
        self.deconv3.weight = nn.Parameter(cnn.conv3.weight.data)
        self.deconv2.weight = nn.Parameter(cnn.conv2.weight.data)
        self.deconv1.weight = nn.Parameter(cnn.conv1.weight.data)

    def forward(self, x, pool_info):
        indices3 = pool_info['pool3']['indices']
        size3 = pool_info['pool3']['output_size']
        x = self.unpool3(x, indices3, output_size=size3)
        x = self.relu3(x)
        x = self.deconv3(x)
        # x is the feature map from the last convolutional block (before flattening)
        # Unpool the second pooling operation
        indices2 = pool_info['pool2']['indices']
        size2 = pool_info['pool2']['output_size']
        x = self.unpool2(x, indices2, output_size=size2)
        x = self.relu2(x)
        x = self.deconv2(x)
        
        # Unpool the first pooling operation
        indices1 = pool_info['pool1']['indices']
        size1 = pool_info['pool1']['output_size']
        x = self.unpool1(x, indices1, output_size=size1)
        x = self.relu1(x)
        x = self.deconv1(x)
        return x

def train_deconvnet(cnn, deconvnet, dataloader, num_epochs=10, lr=1e-3, device='cpu'):
    cnn.eval()  # CNN is fixed
    deconvnet.train()
    optimizer = optim.Adam(deconvnet.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in tqdm(range(num_epochs)):
        total_loss = 0.0
        for images, _ in dataloader:
            images = images.to(device)
            # Forward pass through CNN.
            _ = cnn(images)
            # For simplicity, take the activations after the second pooling.
            # (In practice you might choose a particular layer.)
            x = cnn.conv1(images)
            x = cnn.relu1(x)
            size1 = x.size()
            x, indices1 = cnn.pool1(x)
            x = cnn.conv2(x)
            x = cnn.relu2(x)
            size2 = x.size()
            x, indices2 = cnn.pool2(x)
            x = cnn.conv3(x)
            x = cnn.relu3(x)
            size3 = x.size()
            x, indices3 = cnn.pool3(x)
            # Save pool information in CNN (so our deconvnet can use it)
            cnn.pool_info = {
                'pool1': {'indices': indices1, 'output_size': size1},
                'pool2': {'indices': indices2, 'output_size': size2},
                'pool3': {'indices': indices3, 'output_size': size3}
            }
            # Get the reconstruction from the deconvnet.
            reconstruction = deconvnet(x, cnn.pool_info)
            loss = criterion(reconstruction, images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.4f}")

def visualize_activation(cnn, deconvnet, input_image, layer='conv2', filter_index=0):
    """
    This function visualizes what a particular filter (specified by filter_index)
    in a given convolutional layer (here 'conv2') is looking for.
    
    The process is:
      a. Forward propagate the image through the CNN, storing pooling indices.
      b. In the chosen layer's output, zero out all activations except for the maximum
         response in the selected filter.
      c. Use the deconvnet to backproject the activation into pixel space.
      d. Visualize the resulting “reconstruction”.
    """
    cnn.eval()
    deconvnet.eval()
    
    # Ensure the input image requires gradients (if needed for more complex schemes)
    input_image.requires_grad = True
    
    # Forward pass through the CNN.
    # We ignore the final classification output.
    _ = cnn(input_image)
    
    # Get the feature map for the selected layer.
    # In this example, we assume we want the activations from conv2 (after ReLU, before pooling).
    # One way is to register a forward hook. For brevity, let’s assume we can capture it here.
    # (A more complete implementation would register a hook on cnn.conv2.)
    with torch.no_grad():
        # Forward pass up to conv2:
        x = cnn.conv1(input_image)
        x = cnn.relu1(x)
        size1 = x.size()
        x, indices1 = cnn.pool1(x)
        x = cnn.conv2(x)
        x = cnn.relu2(x)
        x, indices2 = cnn.pool2(x)
        x = cnn.conv3(x)
        x = cnn.relu3(x)
        # x now has shape [batch, 16, H, W]
    
    # Select the filter of interest.
    feature_map = x[0, filter_index, :, :]  # take the first image in the batch
    # Find the position of maximum activation
    max_val = feature_map.max()
    mask = torch.zeros_like(x)
    # Compute spatial indices:
    idx = feature_map.argmax()
    h, w = feature_map.shape
    row = idx // w
    col = idx % w
    mask[0, filter_index, row, col] = max_val

    # Now, perform a backward pass through the deconvnet.
    # In the standard deconvnet visualization, you “invert” the operations.
    # For our deconvnet, we need to supply the feature map corresponding to conv2’s output.
    # Here we simulate the process by backprojecting from the masked activation.
    # Note: In practice one might modify the backward pass (e.g., with custom ReLU hooks)
    # to allow only positive gradients. Here we use the deconvnet forward for reconstruction.
    # First, we need to get the output of the second pooling layer.
    # Continue the forward pass through conv2's pooling:
    x_pooled, indices3 = cnn.pool3(x)
    # Replace the feature map with the mask (we assume the mask is applied after conv2 and before pool2)
    # x_pooled = mask  # This isolates the single activation.
    
    # Now, use the deconvnet to reconstruct the image.
    # Pass the masked feature map along with stored pool information.
    reconstruction = deconvnet(x_pooled, cnn.pool_info)
    
    # Normalize the reconstruction for visualization.
    rec = reconstruction.detach().cpu().squeeze()
    rec = rec - rec.min()
    if rec.max() > 0:
        rec = rec / rec.max()
    
    # Display the result.
    plt.figure(figsize=(3, 3))
    plt.imshow(rec.numpy(), cmap='gray')
    plt.title(f"{layer} - Filter {filter_index}")
    plt.axis('off')
    plt.show()

    if __name__ == "__main__":
        model = get_trained_cnn()
        dataset, _, _ = get_datasets()
        inp = dataset[0]
        print(model)
        print(inp[0].shape)
        # currently doesn't work because of a shape problem
        print(model(inp[0]))
