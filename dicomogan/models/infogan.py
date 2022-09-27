import torch
import torch.nn as nn

# Code taken from Supervised Attribute Information Removal and Reconstruction for Image Manipulation paper
# https://github.com/NannanLi999/AIRR/blob/main/model_attr.py

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        
        self.block = nn.Sequential([
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(dim)
        ])
        
    def forward(self, x):
        h = self.block(x)
        h += x
        return h

def down_sampling(input_nc, output_nc):     
        h = nn.Sequential(nn.Conv2d(input_nc, output_nc, kernel_size=3, stride=2,padding=1, bias=False),
                          nn.BatchNorm2d(output_nc),#Norm()(h)
                          nn.ReLU()
                          )
        return h

class Classifier(nn.Module):
    def __init__(self, dim, num_classes):
        super(Classifier, self).__init__()
        
        self.net = nn.Sequential([
            ResidualBlock(dim),
            down_sampling(dim, 2*dim),
            nn.AvgPool2d(14),
            nn.Conv2d(2*dim, num_classes, kernel_size=1, bias=True),
        ])
    
    def forward(self, x):
        return self.net(x).squeeze(3).squeeze(2)
    
# TODO: Do we need embeddings for each class, or is the one_hot vector enough?
# TODO: Do we need a separate discriminator for this loss, or do we include it with the existing discriminator in 
# the other file
# TODO: What is the difference between encoder classifier and regular classifier? Does the method require a classifier
# at both places, or can we just use the original classifier