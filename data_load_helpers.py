import torch
import torchvision
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import Omniglot
from torch.utils.data import DataLoader, Subset
from PIL import Image

class customized_transform:
    def __call__(self, img):
        if img.mode == 'L':  # 'L' mode is for grayscale images in PIL
            img = img.convert('RGB')  # Convert grayscale to RGB
        # For both grayscale (converted to RGB) and RGB images, apply the same transformation
        return transforms.Compose([
            transforms.Resize((224, 224)),  # Resize the image
            transforms.ToTensor()  # Convert image to tensor (3, 224, 224)
        ])(img)

def prepare_data_for_sklearn(fsl_loader):
    # Initialize lists to store training and testing data
    support_dataset = []
    query_dataset = []
    
    # Iterate through the batches of the test_loader
    for data, labels in fsl_loader:
        # Check if the batch size is larger than 5
        if len(data) > 5:
            # Split the batch into training and testing data
            support_data = data[:5]
            query_data = data[5:]
            support_labels = labels[:5]
            query_labels = labels[5:]
            
            # Append to the respective lists
            support_dataset.append((support_data, support_labels))
            query_dataset.append((query_data, query_labels))
        else:
            # If batch size is less than or equal to 5, just append the entire batch
            support_dataset.append((data, labels))
            query_dataset.append((torch.empty(0), torch.empty(0)))  # Empty test data
    
    # Optionally, concatenate the lists into single tensors
    pre_support_data_tensor = torch.cat([x[0] for x in support_dataset], dim=0)
    support_data_tensor = flatten_images(pre_support_data_tensor).numpy()
    
    support_labels_tensor = torch.cat([x[1] for x in support_dataset], dim=0).numpy()
    pre_query_data_tensor = torch.cat([x[0] for x in query_dataset], dim=0)
    query_data_tensor = flatten_images(pre_query_data_tensor).numpy()
    query_labels_tensor = torch.cat([x[1] for x in query_dataset], dim=0).numpy()
    
    return support_data_tensor, support_labels_tensor, query_data_tensor, query_labels_tensor
    
def flatten_images(data_tensor):
    # Flatten the images from [batch_size, channels, height, width] to [batch_size, features]
    return data_tensor.view(data_tensor.size(0), -1)

