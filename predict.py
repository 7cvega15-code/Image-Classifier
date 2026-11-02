import argparse
import json
from collections import OrderedDict

import numpy as np
from PIL import Image
import torch
from torch import nn
from torchvision import models


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('true', '1', 'yes'):
        return True
    if value.lower() in ('false', '0', 'no'):
        return False
    raise argparse.ArgumentTypeError('Use true or false.')

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
    # Image preprocessing is performed once inside predict().
   
    topk_probability, topk_classes, topk_flowers= predict(args.image, model, device, args.cat_names, args.top_k)

  
    print('Flower: ', topk_flowers)
    print('Probablities: ', topk_probability)
    print('Top classes: ', topk_classes)
    print('Predict complete')
    
    
def get_input_args():
    print('input args')
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoint.pth', help='Load saved model')    
    parser.add_argument('--image', type=str, default='flowers/test/1/image_06764.jpg', help='Process images')   
    parser.add_argument('--gpu', type=parse_bool, nargs='?', const=True, default=False, help='Use CUDA if available; defaults to CPU')  
    parser.add_argument('--top_k', type=int, default=5, help='top 5')
    parser.add_argument('--cat_names', type=str, default='cat_to_name.json', help='Mapping of category  names')
  
    print('input args complete')
    return parser.parse_args()


# load the checkpoint and rebuild the model 
def load_checkpoint(checkpoint):
    # Restricted loading: checkpoints contain tensors and basic metadata only.
    # Never retry a rejected legacy pickle with weights_only=False.
    checkpoint = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if checkpoint.get('format_version') == 3:
        if checkpoint.get('arch') != 'resnet18':
            raise ValueError('Unsupported architecture for checkpoint format 3')
        from resnet_classifier import build_resnet18
        model = build_resnet18(checkpoint['hidden_units'], checkpoint['dropout'], checkpoint['num_classes'])
        model.load_state_dict(checkpoint['state_dict'])
        model.class_to_idx = checkpoint['class_to_idx']
        return model
    if checkpoint.get('format_version') != 2:
        raise ValueError('Unsupported legacy checkpoint. Retrain with this version; do not enable unsafe pickle loading.')
    arch = checkpoint['arch']
    if arch not in ('vgg16', 'alexnet'):
        raise ValueError('Unsupported architecture in checkpoint')
    model = getattr(models, arch)(weights=None)
    inputs = next(layer.in_features for layer in model.classifier if isinstance(layer, nn.Linear))
    hidden = checkpoint['hidden_units']
    model.classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(inputs, hidden)),
        ('relu', nn.ReLU()),
        ('dropout1', nn.Dropout(p=0.5)),
        ('fc2', nn.Linear(hidden, 500)),
        ('relu2', nn.ReLU()),
        ('dropout2', nn.Dropout(p=0.5)),
        ('fc3', nn.Linear(500, 102)),
        ('output', nn.LogSoftmax(dim=1)),
    ]))
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model

# TODO: Process a PIL image for use in a PyTorch model
def process_image(image):
    print('start process image')
    #open the image
    with Image.open(image) as source:
        im = source.convert('RGB')

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

    return np_image.astype(np.float32)

def predict(image_path, model, device, cat_to_name_file, topk):
   
    print('start predict function')
    if not 1 <= topk <= len(model.class_to_idx):
        raise ValueError('top_k must be between 1 and the number of known categories')
    model.to(device) 
    model.eval()
   
    # Process image
    if getattr(model, 'architecture', None) == 'resnet18':
        from resnet_classifier import image_transform
        with Image.open(image_path) as source:
            image_tensor = image_transform()(source.convert('RGB'))
    else:
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
     
