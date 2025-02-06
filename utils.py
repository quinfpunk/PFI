import os
import zipfile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
import matplotlib.pyplot as plt

def get_datasets():
    """
        Returns the training, validation, and test datasets.
    """
    data_dir = "ships_dataset"

    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize images for CNN input
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, "valid"), transform=transform)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=transform)

    return train_dataset, val_dataset, test_dataset



def get_data_loaders(train_dataset, val_dataset, test_dataset):
    """
        Returns DataLoader objects for training, validation, and test sets.

        Args:
            train_dataset (Dataset): Training dataset
            val_dataset (Dataset): Validation dataset
            test_dataset (Dataset): Test dataset
    """
    batch_size = 32

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader



def plot_losses_model(avg_train_losses, avg_val_losses, epochs=10):
    """
        Plots the average training and validation losses over epochs.

        Args:
            avg_train_losses (list): List of average training losses for each epoch
            avg_val_losses (list): List of average validation losses for each epoch
            epochs (int): Number of epochs
    """
    plt.figure(figsize=(8,5))
    plt.plot(range(epochs), avg_train_losses, label='Training Loss', marker='o', linestyle='-')
    plt.plot(range(epochs), avg_val_losses, label='Validation Loss', marker='s', linestyle='--')

    # Labels and Title
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()



def train_save_model(model, train_loader, val_loader, epochs=10, lr=0.001, save_path="ship_model.pth"):
    """
        Trains a PyTorch model and saves the best model based on validation loss.

        Args:
            model (nn.Module): PyTorch model to train
            train_loader (DataLoader): DataLoader for training set
            val_loader (DataLoader): DataLoader for validation set
            epochs (int): Number of epochs to train the model
            lr (float): Learning rate for the optimizer
            save_path (str): Path to save the best model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Set device to CUDA if available
    model.to(device)  # Move model to CUDA if available

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []  
    val_losses = []  
    best_val_loss = float("inf") 
    avg_train_losses = []
    avg_val_losses = []

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss_epoch = []  # Store per-batch losses for this epoch

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Training]", leave=False):
            images, labels = images.to(device), labels.to(device)  # Move images and labels to the device (GPU)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss_epoch.append(loss.item())  # Append the loss of the current batch

        # Compute the average loss for this epoch
        avg_train_loss = sum(train_loss_epoch) / len(train_loss_epoch)
        avg_train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss_epoch = []  # Store per-batch losses for validation
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Validation]", leave=False):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss_epoch.append(loss.item())

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Compute the average validation loss for this epoch
        avg_val_loss = sum(val_loss_epoch) / len(val_loss_epoch)
        avg_val_losses.append(avg_val_loss)

        accuracy = 100 * correct / total if total > 0 else 0
        print(f"Epoch {epoch+1}/{epochs} | Avg Train Loss: {avg_train_loss:.4f} | Avg Val Loss: {avg_val_loss:.4f} | Accuracy: {accuracy:.2f}%")

        # Save the model only if validation loss improves
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"✅ Model saved as {save_path} (Best Val Loss: {best_val_loss:.4f})")
    
    plot_losses_model(avg_train_losses, avg_val_losses, epochs)  # Plot average loss history
