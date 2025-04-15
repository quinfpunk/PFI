import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from cnn_model import get_trained_cnn
from torch.autograd import Variable
from utils import get_datasets
from tqdm import tqdm
import random
import time
import torch.nn.functional as F
from collections import defaultdict

def get_active_neurons(conv_layer, activation_layer, input_data):
    with torch.no_grad():
        output = activation_layer(conv_layer(input_data))
        #print(f"Layer {conv_layer}: Output shape = {output.shape}")
        active_neurons = output.abs().sum(dim=[0, 2, 3]) > 0
    return active_neurons, output

def get_aggregated_active_neurons(original_model, dataset, target_class, num_images, thresholds = (0.01,0.01,0.01), batch_size = 32, device = 'cuda' if torch.cuda.is_available() else 'cpu'):
    original_model.to(device)
    original_model.eval()

    target_indices = [i for i, (_,label) in enumerate(dataset) if label == target_class]
    if not target_indices:
        raise ValueError(f"No images found for target class {target_class}")
    
    if num_images < len(target_indices):
        sampled_indices = random.sample(target_indices, num_images)
    else:
        sampled_indices = target_indices
        num_images = len(target_indices)

    print(f"Number of images used for target class {target_class}: {num_images}")

    #Initialize aggregate masks
    agg_active_conv1 = torch.zeros(original_model.conv1.out_channels, dtype=torch.float32, device = device)
    agg_active_conv2 = torch.zeros(original_model.conv2.out_channels, dtype=torch.float32, device = device)
    agg_active_conv3 = torch.zeros(original_model.conv3.out_channels, dtype=torch.float32, device = device)
    total_images_processed = 0

    with torch.no_grad():
        for i in tqdm(range(0, num_images, batch_size), desc = 'Aggregating activations'):
            batch_indices = sampled_indices[i:min(i+batch_size, num_images)]
            current_batch_size = len(batch_indices)
            if current_batch_size == 0 : continue
            batch_images = torch.stack([dataset[idx][0] for idx in batch_indices]).to(device)

            out1_conv = original_model.conv1(batch_images)
            out1_relu = original_model.relu1(out1_conv)
            active1_batch = (out1_relu.abs().sum(dim=[2,3]).sum(dim=0)) # Check if any neuron is active in the batch, should be of shape [conv1.out_channels]
            agg_active_conv1 += active1_batch
            out1_pooled = original_model.pool1(out1_relu)

            out2_conv = original_model.conv2(out1_pooled)
            out2_relu = original_model.relu2(out2_conv)
            active2_batch = (out2_relu.abs().sum(dim=[2,3]).sum(dim=0)) # Check if any neuron is active in the batch, should be of shape [conv2.out_channels]
            agg_active_conv2 += active2_batch
            out2_pooled = original_model.pool2(out2_relu)

            out3_conv = original_model.conv3(out2_pooled)
            out3_relu = original_model.relu3(out3_conv)
            active3_batch = (out3_relu.abs().sum(dim=[2,3]).sum(dim=0)) # Check if any neuron is active in the batch, should be of shape [conv3.out_channels]
            agg_active_conv3 += active3_batch

            total_images_processed += current_batch_size

    mean_act_conv1 = agg_active_conv1.float() / total_images_processed
    mean_act_conv2 = agg_active_conv2.float() / total_images_processed
    mean_act_conv3 = agg_active_conv3.float() / total_images_processed

    if isinstance(thresholds, (float, int)):
        thresh1, thresh2, thresh3 = thresholds, thresholds, thresholds
    elif isinstance(thresholds, (float, tuple)) and len(thresholds) == 3:
        thresh1, thresh2, thresh3 = thresholds
    else:
        raise ValueError("Thresholds should be a float or a tuple of three floats")
    
    final_active_conv1 = mean_act_conv1 > thresh1
    final_active_conv2 = mean_act_conv2 > thresh2
    final_active_conv3 = mean_act_conv3 > thresh3

    num_keep1 = final_active_conv1.sum().item()
    num_keep2 = final_active_conv2.sum().item()
    num_keep3 = final_active_conv3.sum().item()

    return final_active_conv1, final_active_conv2, final_active_conv3

class PrunedCNN(nn.Module):
    def __init__(self, original_model, img_height=128, img_width=128, active_neurons_conv1=None, active_neurons_conv2=None, active_neurons_conv3=None):
        super(PrunedCNN, self).__init__()

        original_device = next(original_model.parameters()).device
        active_neurons_conv1 = active_neurons_conv1.to(original_device)
        active_neurons_conv2 = active_neurons_conv2.to(original_device)
        active_neurons_conv3 = active_neurons_conv3.to(original_device)
        
        #Layer 1
        start_time_l1 = time.time()
        print("Creating conv1")
        num_active1 = active_neurons_conv1.sum().item()
        if num_active1 == 0: print("No active neurons in conv1")
        self.conv1 = nn.Conv2d(in_channels=original_model.conv1.in_channels,
                               out_channels=max(num_active1, 1),
                               kernel_size=original_model.conv1.kernel_size, stride=original_model.conv1.stride,
                               padding=original_model.conv1.padding, bias = (original_model.conv1.bias is not None))
        
        if num_active1 > 0:
            print("Copying conv1 weights")
            self.conv1.weight.data = original_model.conv1.weight[active_neurons_conv1]
            if original_model.conv1.bias is not None:
                self.conv1.bias.data = original_model.conv1.bias[active_neurons_conv1]
        self.relu1 = original_model.relu1
        self.pool1 = original_model.pool1
        print(f"conv1 created in {time.time() - start_time_l1:.2f} seconds")

        start_time_l2 = time.time()
        print("Creating conv2")
        num_active2 = active_neurons_conv2.sum().item()
        if num_active2 == 0: print("No active neurons in conv2")
        in_channels_conv2 = max(num_active1, 1) if num_active1 > 0 else original_model.conv2.in_channels
        self.conv2 = nn.Conv2d(in_channels=in_channels_conv2, 
                               out_channels=max(num_active2, 1),
                               kernel_size=original_model.conv2.kernel_size, stride=original_model.conv2.stride,
                               padding=original_model.conv2.padding, bias = (original_model.conv2.bias is not None))
        
        if num_active2 > 0 and num_active1 > 0:
            print("Copying conv2 weights")
            active_indices_conv1 = torch.where(active_neurons_conv1)[0] # Get the indices of active neurons in conv1
            self.conv2.weight.data = original_model.conv2.weight[active_neurons_conv2][:, active_indices_conv1]
            if original_model.conv2.bias is not None:
                self.conv2.bias.data = original_model.conv2.bias[active_neurons_conv2]
        self.relu2 = original_model.relu2
        self.pool2 = original_model.pool2
        print(f"conv2 created in {time.time() - start_time_l2:.2f} seconds")


        start_time_l2 = time.time()
        print("Creating conv3")
        num_active3 = active_neurons_conv3.sum().item()
        if num_active3 == 0: print("No active neurons in conv3")
        in_channels_conv3 = max(num_active2, 1) if num_active2 > 0 else original_model.conv3.in_channels
        self.conv3 = nn.Conv2d(in_channels=in_channels_conv3, 
                               out_channels=max(num_active3, 1),
                               kernel_size=original_model.conv3.kernel_size, stride=original_model.conv3.stride,
                               padding=original_model.conv3.padding, bias = (original_model.conv3.bias is not None))
        
        if num_active3 > 0 and num_active2 > 0:
            print("Copying conv3 weights")
            active_indices_conv2 = torch.where(active_neurons_conv2)[0]
            self.conv3.weight.data = original_model.conv3.weight[active_neurons_conv3][:, active_indices_conv2]
            if original_model.conv3.bias is not None:
                self.conv3.bias.data = original_model.conv3.bias[active_neurons_conv3]
        self.relu3 = original_model.relu3
        self.pool3 = original_model.pool3
        print(f"conv3 created in {time.time() - start_time_l2:.2f} seconds")

        #Dummy example to know what size fc1 needs to be
        if num_active1 > 0 and num_active2 > 0 and num_active3 > 0:
            dummy_input = torch.zeros(1, original_model.conv1.in_channels, img_height, img_width, device = original_device)
            with torch.no_grad():
                dummy_out = self.pool3(self.relu3(self.conv3(
                    self.pool2(self.relu2(self.conv2(
                    self.pool1(self.relu1(self.conv1(dummy_input)))))))))
            flattened_size = dummy_out.view(1, -1).shape[1]
        else:
            print("One of the layers has no active neurons, using default size")
            flattened_size = 1


        self.fc1 = nn.Linear(flattened_size, 128)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()
        print(f"fc1 created in {time.time() - start_time_l2:.2f} seconds")

        #Note, this does mean that the fc layers are randomly initialized.... We might want to copy parts of the original fc1 and fc2, but seems hard to do...

    def forward(self, x):
        if self.conv1.out_channels == 0:
            raise ValueError("No active neurons in conv1")
        x = self.pool1(self.relu1(self.conv1(x)))
        if self.conv2.out_channels == 0:
            raise ValueError("No active neurons in conv2")
        x = self.pool2(self.relu2(self.conv2(x)))
        if self.conv3.out_channels == 0:
            raise ValueError("No active neurons in conv3")
        x = self.pool3(self.relu3(self.conv3(x)))
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return self.sigmoid(x)

if __name__ == "__main__":
    target_class = 0
    num_images_for_pruning = 20
    pruning_batch_size = 16
    absolute_thresholds = (0.05, 0.05, 0.05)
    dataset_example_number = 5058
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    

    original_model = get_trained_cnn()
    original_model.to(device)
    dataset, _, _ = get_datasets()
    sample_image = dataset[dataset_example_number][0].unsqueeze(0).to(device)
    print("Target class: " + str(target_class))
    print(f"Dataset image example class : {dataset[dataset_example_number][1] + 1}")
    

    active_n1, active_n2, active_n3 = get_aggregated_active_neurons(original_model, dataset, target_class, num_images_for_pruning, batch_size = pruning_batch_size, device= device)
    img_height, img_width = dataset[0][0].shape[1], dataset[0][0].shape[2]
    
    
    pruned_model = PrunedCNN(original_model, img_height=img_height, img_width=img_width, active_neurons_conv1=active_n1, active_neurons_conv2=active_n2, active_neurons_conv3=active_n3)
    pruned_model.to(device)
    print("Pruned model created")
    print(pruned_model)

    torch.save(pruned_model.state_dict(), f"pruned_model_{target_class +1}.pth")
    print("Pruned model saved")

    output = pruned_model(sample_image)
    print("Example output:", output)

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
    """"""
    def create_multiclass_batch(dataset, num_classes, samples_per_class, device):
        indices_by_class = defaultdict(list)

        targets = None
        if hasattr(dataset, 'targets'):
            targets = dataset.targets
        elif hasattr(dataset, 'labels'):
            targets = dataset.labels

        if targets is not None:
            if isinstance(targets, torch.Tensor):
                targets = targets.cpu().tolist()
            for i, label in enumerate(targets):
                indices_by_class[label].append(i)
        else:
            for i in tqdm(range(len(dataset)), desc = "Scanning Dataset"):
                try:
                    _, label = dataset[i]
                    if isinstance(label, torch.Tensor): label = label.item()
                    indices_by_class[label].append(i)
                except Exception as e:
                    print(f"Error processing index {i}: {e}")
                    continue #Skip this item

        expected_classes = list(range(num_classes))
        found_classes = sorted(indices_by_class.keys())
        if len(found_classes) < num_classes:
            raise ValueError(f"Not enough classes in the dataset. Found: {found_classes}, expected: {num_classes}")
        selected_indices = []
        selected_labels = []
        
        #Sample indices
        for class_label in expected_classes:
            if class_label not in indices_by_class:
                raise ValueError(f"Class {class_label} not found in dataset. Found classes: {found_classes}")
            
            class_indices = indices_by_class[class_label]
            if len(class_indices) < samples_per_class:
                raise ValueError(f"Not enough samples for class {class_label}. Found: {len(class_indices)}, expected: {samples_per_class}")
            else:
                sampled = random.sample(class_indices, samples_per_class)
                samples_to_add = samples_per_class

            selected_indices.extend(sampled)
            selected_labels.extend([class_label] * samples_to_add)


        #Retrieve the image tensors
        image_tensors = []
        retrieved_labels = []
        for idx in tqdm(selected_indices, desc = "Retrieving images"):
            try:
                image_tensor, label = dataset[idx]
                image_tensors.append(image_tensor)
                
            except Exception as e:
                raise RuntimeError(f"Error retrieving image at index {idx}: {e}")
            


        final_batch_images = torch.stack(image_tensors).to(device)
        final_batch_labels = torch.tensor(selected_labels, dtype = torch.long).to(device)


        return final_batch_images, final_batch_labels
    

#Create a batch of images
    sample_image_batch, sample_labels_batch = create_multiclass_batch(dataset, num_classes = 10, samples_per_class = 1, device = device)
    print(f"Sample batch created with shape: {sample_image_batch.shape}")

    """
    #Define the saliency map calculation
    def compute_saliency_map(model, image_batch_in):
        model.eval()
        #print(f"Image shape: {image.shape}")
        #print(f"Image values: {image}")
        
        if image_batch_in.dim() == 3:
            image_batch_processed = image_batch_in.unsqueeze(0)  # Add batch dimension if missing
        elif image_batch_in.dim() == 4:
            image_batch_processed = image_batch_in
        else:
            raise ValueError("Unexpected image dimensions")
        
        image_batch_processed.requires_grad_(True)
        output = model(image_batch_processed)


        model.zero_grad() # Clear previous gradients
        if image_batch_processed.grad is not None:
            image_batch_processed.grad.zero_()
        
        if output.shape[1] > 1: #Original model, numel gets the number of elements in a tensor
            target_score = output.max(dim=1)[0] # Get the maximum score for each image in the batch
        elif output.shape[1]==1: #Pruned model, output is a single value
            target_score = output.squeeze(-1) # Remove the last dimension
        else:
            raise ValueError("Unexpected output shape")

        target_score.sum().backward() # Backpropagate to get gradients

        if image_batch_processed.grad is None:
            raise ValueError("Gradients are None, check the model and input data.")
        gradient = image_batch_processed.grad # Get the gradients of the input image
        
        saliency, _ = torch.max(gradient.abs(), dim=1) # Get the maximum gradient across the color channels
        return saliency.cpu().detach().numpy() # Convert to numpy array for visualization


    #Compute saliency maps for the original and pruned models
    import matplotlib.pyplot as plt

    saliency_original_batch = compute_saliency_map(original_model, sample_image_batch)
    saliency_pruned_batch = compute_saliency_map(pruned_model, sample_image_batch)

    batch_size = sample_image_batch.shape[0]
    for i in range(batch_size):
        img_tensor = sample_image_batch[i]
        label = sample_labels_batch[i].item()
    
        saliency = saliency_original_batch[i]
        saliency2 = saliency_pruned_batch[i]

        plt.figure(figsize=(12, 4))
        plt.suptitle(f'Saliency Maps - Pruned on model {target_class + 1}')
        
        plt.subplot(1, 3, 1)
        plt.imshow(img_tensor.cpu().permute(1, 2, 0).detach().numpy())
        plt.title(f'Input image (Class {label + 1})')
        plt.axis('off')
    
        plt.subplot(1, 3, 2)
        plt.imshow(saliency, cmap='hot')
        plt.title('Original model saliency map')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(saliency2, cmap='hot')
        plt.title('Pruned model saliency map')
        plt.axis('off')
        plt.tight_layout(rect = [0, 0.03, 1, 0.95])

    plt.show()

    """
    import matplotlib.pyplot as plt
    import numpy as np
    from torch.autograd import Function
    import torch.nn.functional as F
    import cv2

    class GradCAM:
        def __init__(self, model, target_layer):
            self.model = model
            self.target_layer = target_layer
            self.gradients = None
            self.activation_maps = None
            self._device = next(model.parameters()).device
            self.hook_handles = []

        def _register_hooks(self):
            self._remove_hooks()
            handle_bwd = self.target_layer.register_full_backward_hook(self._save_gradient)
            handle_fwd = self.target_layer.register_forward_hook(self._hook)
            self.hook_handles.extend([handle_fwd, handle_bwd])

        def _save_gradient(self, module, grad_input, grad_output):
            if grad_output[0] is not None:
                self.gradients = grad_output[0].detach()
            else:
                print("Got None gradient in GradCAM backward hook")

        def _hook(self, module, input, output):
            self.activation_maps = output.detach()
        
        def _remove_hooks(self):
            for handle in self.hook_handles:
                handle.remove()
            self.hook_handles = []

        def generate_cam(self, input_image_batch, target_class_index = None):
            self.model.eval()
            input_image_batch = input_image_batch.clone().detach().to(self._device) # Move to the same device as the model

            self._register_hooks()

            output = self.model(input_image_batch)
            #print(f"Output shape: {output.shape}, Output values: {output}")
            
            if output.ndim == 1:
                output = output.unsqueeze(1)
            

            if output.shape[1] > 1: #Original model
                if target_class_index is None:
                    target_class_index = output.argmax(dim=1) #Shape [batch_size]
                elif isinstance(target_class_index, int):
                    target_class_index = torch.tensor([target_class_index] * input_image_batch.shape[0], device=self._device)
                elif isinstance(target_class_index, torch.Tensor):
                    target_class_index = target_class_index.to(dtype=torch.long, device=self._device)

                if target_class_index.dim() == 0: target_class_index = target_class_index.unsqueeze(0) # Add batch dimension if missing

                score = torch.gather(output, 1, target_class_index.unsqueeze(-1)).squeeze(-1) # Gather the scores for the target class

            elif output.shape[1] == 1: #Pruned model
                score = output.squeeze(-1)
            else:
                self._remove_hooks()
                raise ValueError("Unexpected model output shape")
            
            
            self.model.zero_grad()
            score.sum().backward()

            if self.gradients is None or self.activation_maps is None:
                self._remove_hooks()
                grad_status = 'OK' if self.gradients is not None else 'None'
                act_status = 'OK' if self.activation_maps is not None else 'None'
                raise RuntimeError(f"Gradients ({grad_status}) or activation maps ({act_status}) are None.")
            
            pooled_gradients = torch.mean(self.gradients, dim=[2,3])
            activations = self.activation_maps

            pool_gradients_resized = pooled_gradients.unsqueeze(-1).unsqueeze(-1)
            
            activations *= pool_gradients_resized

            heatmap = torch.mean(activations, dim=1)
            heatmap = F.relu(heatmap)

            batch_size = heatmap.shape[0]
            normalized_heatmap = torch.zeros_like(heatmap)
            for i in range(batch_size):
                img_heatmap = heatmap[i]
                max_val = torch.amax(img_heatmap)
                min_val = torch.amin(img_heatmap)
                normalized_heatmap[i] = (img_heatmap - min_val) / (max_val - min_val + 1e-10)

            self._remove_hooks()

            return normalized_heatmap

        def overlay_heatmap(self, image, heatmap, alpha=0.5):
            #The gradCAM does not keep the original size, so I will upscale it
            original_height, original_width = image.shape[1], image.shape[2]
            new_heatmap = F.interpolate(heatmap.unsqueeze(0).unsqueeze(0), size=(original_height, original_width), mode='bilinear', align_corners=False).squeeze()

            #Normalize to 0-1
            new_heatmap = new_heatmap - new_heatmap.min()
            new_heatmap = new_heatmap / new_heatmap.max()

            #Convert to RGB
            new_heatmap = np.uint8(255 * new_heatmap.cpu().detach().numpy())
            new_heatmap = cv2.applyColorMap(new_heatmap, cv2.COLORMAP_TURBO)
            new_heatmap = cv2.cvtColor(new_heatmap, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
            new_heatmap = np.float32(new_heatmap) / 255

            # And now we convert the original image to numpy as well
            image = image.squeeze().permute(1, 2, 0).cpu().detach().numpy()

            superimposed_image = np.float32(image) + np.float32(new_heatmap)
            superimposed_image = np.clip(superimposed_image, 0, 1)
            return superimposed_image

        
    @staticmethod
    def compute_and__display_gradcam(model, image_tensor_batch, target_layer, target_class=None, is_pruned=False, alpha = 0.5):
        try:
            if len(image_tensor_batch.shape) == 3:
                image_tensor_batch = image_tensor_batch.unsqueeze(0)  # Add batch dimension if missing
            elif len(image_tensor_batch.shape) == 4:
                pass
            else:
                raise ValueError("Unexpected image dimensions")
            
            batch_size = image_tensor_batch.shape[0]
        
            grad_cam = GradCAM(model, target_layer)
            actual_target_idx = target_class if not is_pruned else None
            heatmap = grad_cam.generate_cam(image_tensor_batch, actual_target_idx) #heatmap is actually a batch of heatmaps
            
            for i in range(batch_size):
            
                original_img = image_tensor_batch[i].cpu()
                heatmap_tensor = heatmap[i] # Get the heatmap for the current image
                superimposed_image = grad_cam.overlay_heatmap(original_img, heatmap_tensor, alpha=alpha)

                fig, axes = plt.subplots(1, 3, figsize=(15,5))

                img_display = original_img.permute(1, 2, 0).cpu().detach().numpy()
                if img_display.min() < -0.1 or img_display.max() > 1.1:
                    if img_display.max() > 100: img_display=img_display/255.0
                    elif img_display.min() < -0.1: img_display = (img_display + 1.0) / 2.0
            
            
                axes[0].imshow(np.clip(img_display, 0, 1))
                axes[0].set_title('Original image')
                axes[0].axis('off')

                axes[1].imshow(heatmap_tensor.numpy(), cmap='turbo')
                axes[1].set_title('Grad-CAM Heatmap only')
                axes[1].axis('off')

                axes[2].imshow(superimposed_image, cmap='turbo')
                axes[2].set_title('Grad-CAM Overlay')
                axes[2].axis('off')

                if is_pruned:
                    plt.suptitle('Pruned model')
                else:
                    plt.suptitle('Original model')

                plt.show()

        except Exception as e:
            model_type = "Pruned" if is_pruned else "Original"
            print(f"Error in Grad-CAM computation for {model_type} model: {e}")
            print(f"Error Type : {type(e).__name__}")
            print(f"Error Message : {e}")
            print("Traceback")
            import traceback
            traceback.print_exc()
            print("End traceback")
            return None


    #Evaluate GradCAM for the original and pruned models
    target_layer = original_model.conv3

    print("Evaluating GradCAM for Original Model")
    compute_and__display_gradcam(original_model, sample_image_batch, target_layer, is_pruned=False)

    target_layer = pruned_model.conv3

    print("Evaluating GradCAM for Pruned Model")
    compute_and__display_gradcam(pruned_model, sample_image_batch, target_layer, is_pruned=True)


"""
    #Capture activations for the pruned model
    captured_activations = {}
    def get_activation_hook(layer_name):
        def hook(model, input, output):
            captured_activations[layer_name] = output.detach().cpu()
        return hook
    
    def get_activations(model, target_layer, dataloader, device):
        model.eval()
        model.to(device)
        layer_outputs = []
        handle = None
        layer_name= 'target_layer_activations'

        try:
            handle = target_layer.register_forward_hook(get_activation_hook(layer_name))
            for inputs, _ in tqdm(dataloader, desc="Capturing activations"):
                inputs = inputs.to(device)
                _ = model(inputs) # Forward pass to capture activations

                #Check if hook captured something
                if layer_name in captured_activations:
                    layer_outputs.append(captured_activations[layer_name].clone())
                    del captured_activations[layer_name] 
                else:
                    print(f"Layer {layer_name} did not capture any activations.")

        except Exception as e:
            print(f"Error in capturing activations: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            #REMOVE THE HOOOOK"
            if handle:
                handle.remove()
            captured_activations.clear()

        if not layer_outputs:
            print("No activations captured.")
            return None
        

        #Concatenate activations from all batches
        try:
            all_activations = torch.cat(layer_outputs, dim=0)
            print(f"Successfully captured activations shape: {all_activations.shape}")
            return all_activations
        except Exception as e:
            print(f"Error concatenating activations: {e}")
            return None
    """