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
        
        self.net = nn.Sequential(
            ResidualBlock(dim),
            down_sampling(dim, 2*dim),
            nn.AvgPool2d(14),
            nn.Conv2d(2*dim, num_classes, kernel_size=1, bias=True),
        )
    
    def forward(self, x):
        return self.net(x).squeeze(3).squeeze(2)

class EncoderClassifier(nn.Module):
    def __init__(self, dim, num_classes):
        super(EncoderClassifier, self).__init__()

        self.net = nn.Sequential(
            down_sampling(dim, 2*dim),
            nn.AvgPool2d(14),
            nn.Conv2d(2*dim, num_classes, kernel_size=1, bias=True)
        )
    
    def forward(self, x):
        return self.net(x).squeeze(3).squeeze(2)

# TODO: Figure out what the format of num_classes needs to be
# TODO: Test this loss with proper vectors
class EncoderLoss(nn.Module):
    def __init__(self, mse_reduction, sce_reduction, lsm_dim, classifier, num_classes):
        super(EncoderLoss, self).__init__()

        self.mse_loss = torch.nn.MSELoss(reduction=mse_reduction)
        self.sce_loss = torch.nn.SCELoss(reduction=sce_reduction)
        self.logsoftmax = torch.nn.LogSoftmax(dim=lsm_dim)

        self.classifier = classifier
        self.num_classes = num_classes

    def forward(self, x, attr_vec):
        # Step 1: Compute logits from x using classifier
        # Step 2: Compute loss compared to y

        logits = self.classifier(x)

        cur_index = 0
        attr_loss = 0

        for i in range(len(self.num_classes)):
            alogits=logits[:,cur_index:cur_index+self.num_classes[i]]
            aloss,_=torch.max(self.logsoftmax(alogits),1)
            aloss=torch.maximum(aloss-torch.log(torch.ones_like(aloss)/self.num_classes[i]),1e-2*torch.ones_like(aloss))
            attr_loss+=aloss.mean()
            cur_index+=self.num_classes[i]
        return attr_loss
