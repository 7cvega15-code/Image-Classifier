# Imports here

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models
import torch.optim as optim
import numpy as np
from collections import OrderedDict
from PIL import Image
from torch.autograd import Variable
import argparse
import os
import sys
import pandas as pd
import json


#start predict.py
def main():
    print('start main')
    args = get_input_args()
    is_gpu = args.gpu
    use_cuda = torch.cuda.is_available()
    device = torch.device("cpu")
  
    if is_gpu and use_cuda:
         device = torch.device("cuda")
         print(f"Device is set to {device}")

    else:
        device = torch.device("cpu")
        print(f"Device is set to {device}")
                  
    print('call load_checkpoint')              
    model = load_checkpoint(args.checkpoint)
    #modify n_image name?? – check if it runs
    n_image = process_image(args.image)
   
    topk_probability, topk_classes, topk_flowers= predict(args.image, model, device, args.cat_names, args.top_k)

  
    print('Flower: ', topk_flowers)
    print('Probablities: ', topk_probability)
    print('Top 5 classes : ', topk_classes)
    print('Predict complete')
    
    
def get_input_args():
    print('input args')
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoint.pth', help='Load saved model')    
    parser.add_argument('--image', type=str, default='flowers/test/1/image_06764.jpg', help='Process images')   
    parser.add_argument('--gpu', type=bool, default='True', help='True: gpu, False: cpu')  
    parser.add_argument('--top_k', type=int, default=5, help='top 5')
    parser.add_argument('--cat_names', type=str, default='cat_to_name.json', help='Mapping of category  names')
  
    print('input args complete')
    return parser.parse_args()


# load the checkpoint and rebuild the model 
def load_checkpoint(checkpoint):

    checkpoint= torch.load(checkpoint, map_location=lambda storage, loc: storage)
    model = getattr(models, checkpoint['arch'])(pretrained=True)     
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    #load the actual classifier state dict that we added -  include: .classifier - Shinto 3/14
    model.classifier.load_state_dict(checkpoint['state_dict'])
    #don’t need to load the optimizer as we wont continue training for this project       
    #model.optimizer.load_state_dict(checkpoint['optimizer_state'])
    
    return model
    print('Checkpoint loaded')
# TODO: Process a PIL image for use in a PyTorch model
def process_image(image):
    print('start process image')
    #open the image
    im = Image.open(image)

    #resize and crop 
    current_width, current_height = im.size
    if current_width < current_height:
        new_height = int(current_height * 256 /current_width)
        im = im.resize((256, new_height))
    else:
        new_width = int(current_width * 256 / current_height)
        im = im.resize((new_width, 256))
    # crop 224x224
    precrop_width, precrop_height = im.size
    left = (precrop_width - 224)/2
    top = (precrop_height - 224)/2
    right = (precrop_width + 224)/2
    bottom = (precrop_height + 224)/2
    im = im.crop((left, top, right, bottom))
    # make np array and convert the value to a float
    image = np.array(im)
    np_image = np.array(image, dtype = float)
    np_image = image/255
    #normalize with the same mean/std as orig pics
    std = np.array([0.229, 0.224, 0.225])
    mean = np.array([0.485, 0.456, 0.406])

    #subtract the means by each color channel and divide by std
    np_image = (np_image - mean)/std
    #change dimension of the color channel from third to first
    np_image = np_image.transpose((2,0,1))

    return np_image

def predict(image_path, model, device, cat_to_name_file, topk):
   
    print('start predict function')
    model.to(device) 
    model.eval()
   
    # Process image
    img = process_image(image_path)
    image_tensor = torch.from_numpy(img).type(torch.FloatTensor)
   
    # Add batch of size 1 to image to only process 1 pic
    model_input = image_tensor.unsqueeze(0)
    
    # move input image to GPU -Mentor : Shinto 3/8/2019
    model_input = model_input.to(device)
    
    
    with torch.no_grad():
        # get the # Top labels Probability
        output=model.forward(model_input)

        #  return classes & probabilities using topk
        probability, classes = torch.topk(output,topk)
    
        #calculate exponential of probability
        probability = probability.exp().cpu().numpy()[0] 
        classes =  classes.cpu().numpy()[0]
        
        # Convert indices to actual class labels using idx_to_class
        classes = [c for c in classes]
        idx_to_class = {val: key for key, val in    
                        model.class_to_idx.items()}
        
        # Use field classes to get the top_classes
        top_classes = [idx_to_class[l] for l in classes]
        print('top_classes:', top_classes)
        print('probability:', probability)
        
        with open(cat_to_name_file, 'r') as f:
            cat_to_name = json.load(f)
        # Get actual names  for graph plot
        top_flowers = [cat_to_name[t] for t in top_classes]
    
  
    return probability, top_classes, top_flowers
                        
#call main fcn to run the script
if __name__ == '__main__':
     main()     
     
