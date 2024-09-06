import torch
import torchvision
from torchvision import transforms

class protonet(torch.nn.Module):
    def __init__(self, num_classes = 50, embedding = transforms.Lambda(lambda x: x)):
        super().__init__()
        self.prototypes = torch.zeros((num_classes, 1000))
        self.cur_class_size = torch.zeros((num_classes, 1))  
        self.embedding = embedding

    def batch_update_prototype(self, images, labels):
        unique_labels = torch.unique(labels)
        for i in unique_labels:
            idx_to_update = (labels == i)
            x = images[idx_to_update]
            self.prototypes[i] = (self.cur_class_size[i] * self.prototypes[i] + torch.sum(self.embedding(x), dim=0)) / (self.cur_class_size[i] + images.shape[0])
            self.cur_class_size[i] += images.shape[0]
    
    def protonet_fsl(self, x):
        batch_num = x.shape[0]

        num_classes = (self.cur_class_size).shape[0]
        dist_vec = torch.zeros((batch_num, num_classes))
        embeded_x = self.embedding(x)
        for i in range(batch_num):
            for j in range(num_classes):
                dist_vec[i][j] = torch.dist(self.prototypes[j], embeded_x[i], p = 2)

        return dist_vec