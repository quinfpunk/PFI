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
        #print(f"Layer {conv_layer}: Output shape = {output.shape}")
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
    target_class = 5
    dataset_example_number = 5058
    original_model = get_trained_cnn()
    dataset, _, _ = get_datasets()
    print("Target class: " + str(target_class))
    print("Dataset image example class : " + str(dataset[dataset_example_number][1]))
    sample_input = dataset[dataset_example_number][0].unsqueeze(0)
    pruned_model = PrunedCNN(original_model, target_class, sample_input)
    #print(pruned_model)

    output = pruned_model(sample_input)
    print("Output 1st time:", output)

    """ 
    def evaluate_class_performance(model, dataset, is_pruned):
        model.eval()
        class_correct = {i: 0 for i in range(10)}
        class_total = {i: 0 for i in range(10)}
        if is_pruned:
            print("Pruned accuracy Calculation")
        else: 
            print("Original accuracy Calculation")
        with torch.no_grad():
            for image, label in dataset:
                
                outputs = model(image.unsqueeze(0))

                if is_pruned: #1 if tarhet_class, 0 else
                    prediction = (outputs > 0.5).float().item() #convert probability into 0 or 1
                    class_correct[label] += (prediction == 1)  
                    
                else:
                    predicted_class = torch.argmax(outputs, dim=1).item()
                    class_correct[label] += (predicted_class == label)

                class_total[label] += 1
        
        per_class_accuracies = {c: (class_correct[c] / class_total[c]) if class_total[c] > 0 else 0 for c in range(10)}
        overall_accuracy = sum(per_class_accuracies.values()) / 10

        print("\nAccuracies")
        for c in range(10):
            print(f"Class {c}: {per_class_accuracies[c]:.4f}")

        print(f"\nOverall accuracy: {overall_accuracy:.4f}")
        return per_class_accuracies, overall_accuracy

    first_accuracies, first_mean_accuracy = evaluate_class_performance(original_model, dataset, False)
    pruned_accuracies, pruned_mean_accuracy = evaluate_class_performance(pruned_model, dataset, True) 
    """
    def compute_saliency_map(model, image):
        model.eval()
        #print(f"Image shape: {image.shape}")
        #print(f"Image values: {image}")
        image = image.unsqueeze(0).requires_grad_()

        output = model(image)
        
        if output.numel() > 2: #Original model, numel gets the number of elements in a tensor
            predicted_class = output.argmax().item()
            output_scalar = output[0, predicted_class] #select the class output
            #print("Goes in Original model in Saliency Map")
        else: #Pruned model
            output_scalar = output.squeeze()
            #print("Goes in Pruned model in Saliency Map")

        output_scalar.backward()

        saliency = image.grad.abs().squeeze().detach().numpy()
        return saliency

    import matplotlib.pyplot as plt

    sample_image= dataset[dataset_example_number][0]
    saliency = compute_saliency_map(original_model, sample_image)
    saliency2 = compute_saliency_map(pruned_model, sample_image)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(sample_image.permute(1, 2, 0))
    plt.title('Input image')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(saliency.sum(axis=0), cmap='hot')
    plt.title('Original model saliency map')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(saliency2.sum(axis=0), cmap='hot')
    plt.title('Pruned model saliency map')
    plt.axis('off')
    plt.show()

    
    from torch.autograd import Function
    import torch.nn.functional as F
    import cv2

    class GradCAM:
        def __init__(self, model, target_layer):
            self.model = model
            self.target_layer = target_layer
            self.gradients = None
            self.activation_maps = None

        def save_gradient(self, grad):
            self.gradients = grad

        def hook(self, module, input, output):
            self.activation_maps = output
            output.register_hook(self.save_gradient)

        def generate_cam(self, image, gradcam_target_class):
            self.target_layer.register_forward_hook(self.hook) #register hook to target class (related to GradCAM, not pruned model)
            self.model.eval()

            output = self.model(image)
            #print(f"Output shape: {output.shape}, Output values: {output}")
            
            if len(output.shape) == 2: #Pruned model
                if gradcam_target_class is None:
                    gradcam_target_class = 1
                #print(f"Using target class {gradcam_target_class} for Pruned model")
            else: #Original model
                if gradcam_target_class is None:
                    gradcam_target_class = output.argmax().item()
                #print(f"Using target class {gradcam_target_class} for Original model")
            
            target = output if output.shape == torch.Size([1, 1]) else output[0, gradcam_target_class]
            self.model.zero_grad()
            target.backward()

            gradients = self.gradients
            activations = self.activation_maps

            pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
            for i in range(activations.shape[1]):
                activations[:, i, :, :] *= pooled_gradients[i]

            heatmap = torch.mean(activations, dim=1).squeeze()
            heatmap = F.relu(heatmap)
            heatmap /= torch.max(heatmap) #normalize between 0 and 1

            return heatmap

        def overlay_heatmap(self, image, heatmap, alpha=0.5):
            #The gradCAM does not keep the original size, so I will upscale it
            original_height, original_width = image.shape[2], image.shape[3]
            new_heatmap = F.interpolate(heatmap.unsqueeze(0).unsqueeze(0), size=(original_height, original_width), mode='bilinear', align_corners=False).squeeze()

            #Normalize to 0-1
            new_heatmap = new_heatmap - new_heatmap.min()
            new_heatmap = new_heatmap / new_heatmap.max()

            #Convert to RGB
            new_heatmap = np.uint8(255 * new_heatmap.cpu().detach().numpy())
            new_heatmap = cv2.applyColorMap(new_heatmap, cv2.COLORMAP_JET)
            new_heatmap = np.float32(new_heatmap) / 255

            # And now we convert the original image to numpy as well
            image = image.squeeze().permute(1, 2, 0).cpu().detach().numpy()

            superimposed_image = np.float32(image) + np.float32(new_heatmap)
            superimposed_image = np.clip(superimposed_image, 0, 1)
            return superimposed_image

        
    def compute_and__display_gradcam(model, sample_image, target_layer, target_class=None, is_pruned=False):
        grad_cam = GradCAM(model, target_layer)
        heatmap = grad_cam.generate_cam(sample_image, target_class)
        superimposed_image = grad_cam.overlay_heatmap(sample_image, heatmap)

        fig, axes = plt.subplots(1, 2, figsize=(12,6))

        normalized_image = (sample_image + 1) /2
        
        image_to_display = torch.clamp(normalized_image, 0, 1)
        axes[0].imshow(image_to_display.squeeze().cpu().numpy().transpose(1, 2, 0))
        axes[0].set_title('Original image')
        axes[0].axis('off')

        axes[1].imshow(superimposed_image, cmap='hot')
        axes[1].set_title('Grad-CAM')
        axes[1].axis('off')

        if is_pruned:
            plt.suptitle('Pruned model')
        else:
            plt.suptitle('Original model')

        plt.show()
        
        return heatmap

    target_layer = original_model.conv3

    print("Evaluating GradCAM for Original Model")
    compute_and__display_gradcam(original_model, sample_image.unsqueeze(0), target_layer, is_pruned=False)

    target_layer = pruned_model.conv3

    print("Evaluating GradCAM for Pruned Model")
    compute_and__display_gradcam(pruned_model, sample_image.unsqueeze(0), target_layer, is_pruned=True)