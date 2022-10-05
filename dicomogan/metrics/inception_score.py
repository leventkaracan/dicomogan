import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
from PIL import Image
from torchvision.models.inception import inception_v3
import torchvision.transforms as transforms
transform = transforms.Compose([
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) 
import numpy as np
from scipy.stats import entropy

def batch_inception_score(imgs, cuda=True, resize=False, splits=1):
    """Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    #N = len(imgs)

    #assert batch_size > 0
    #assert N > batch_size

    # Set up dtypea
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    #dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)
    #Don't need dataloader since we are only given one batch at this time

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval();
    # TODO: Do we need to modify the size of the input images in order to handle the inception score?
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    #preds = np.zeros((N, 1000))

    #for i, batch in enumerate(dataloader, 0):
    #    batch = batch.type(dtype)
    #    batchv = Variable(batch)
    #    batch_size_i = batch.size()[0]
    #    print(i)
    #    preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    batch = imgs
    batch = batch.type(dtype)
    batchv = Variable(batch)
    pred = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        #part = preds[k * (N // splits): (k+1) * (N // splits), :]
        part = pred
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

def inception_score(imgs, cuda=True, batch_size=1, resize=False, splits=1):
    """Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtypea
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval();
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]
        #print(i)
        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

if __name__ == '__main__':
    images = []
    for i in range(599):
      for y in range(11):
        #images.append(np.asarray(Image.open('/home/levent/sisgan/examples/recon/'+str(i)+'_'+str(y)+'.png')))
        images.append(np.asarray(Image.open('/data/levent/results_fashion_textvaeode2im_adain/recon/'+str(i)+'_'+str(y)+'.png')))
    images = np.array(images).transpose(0,3,1,2)
    images = torch.from_numpy(images) / 255
    images = transform(images)
    
    print(images[0][0])
    print(inception_score(images, resize=True))
