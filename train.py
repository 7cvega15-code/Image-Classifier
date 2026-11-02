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

def main():
   # args = get_input_args()
   # if args.gpu is True:
    #    if torch.cuda.is_available():
     #       device = torch.device("cuda:0")
      #  print('GPU')
            
   # else:
    #    device = torch.device("cpu")
     #   print('CPU')
                    
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
   

    #directories for test, validation and training sets
    data_dir = args.dir 
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    #call load_data fcn 
    train_loader, valid_loader, test_loader, train_data, valid_data, test_data = load_data(train_dir, test_dir, valid_dir)

    #load model 
    model = load_pretrained_network(args.arch)
    model = classifier(model, args.hidden_units)
                     
    #define loss function
    criterion = nn.NLLLoss()

    #define Optimizer
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr)

    model_trained =  train_model(args.epochs, train_loader, valid_loader, device, model, optimizer, criterion)
    validation(model_trained, valid_loader, device, criterion)
    test_model(model_trained, test_loader, device, optimizer, criterion)
    save_checkpoint(args.epochs, model_trained, train_data, args.save_dir, args.arch, optimizer)
    print('training done')


# create parser object and give expected arguments will be used to process the command-#line arguments when the program runs.
def get_input_args(): 
    print('get args')
    parser = argparse.ArgumentParser()                              
    parser.add_argument('--gpu', type=bool, default='True', help='True: gpu, False: cpu')
    parser.add_argument('--arch', type=str, default='vgg16', help='model arch')
    parser.add_argument('--dir', type=str, default='flowers', help='flower images directory path')
    parser.add_argument('--save_dir', type=str, default='checkpoint.pth', help='save model to checkpoint file')
    parser.add_argument('--epochs', type=int, default=5, help='total epochs for the training')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate for the training')
    parser.add_argument('--hidden_units', type=int, default=4000, help='hidden units for classifier')
  
    return parser.parse_args()

                                 
# Define transforms for the training, validation, and testing sets
def load_data(train_dir, test_dir, valid_dir):
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])      

    test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])  

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])  

    # Load the datasets with ImageFolder

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)


    # Using the image datasets and the transforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=32)
    
    return train_loader, valid_loader, test_loader, train_data, valid_data, test_data          
              
print('Data loaded')
              

                  
# Load pre-trained network , choose from at least two different architectures available from torchvision
def load_pretrained_network(arch):
    print('load vgg16 or alexnet model')
    if arch == "vgg16":
         model = models.vgg16(pretrained=True)
         print('Pretrained Network being used is vgg16')
    elif arch == "alexnet":
         model = models.alexnet(pretrained=True)
         print('Pretrained Network being used is alexnet ')

    else:
        print('Invalid : vgg16 will be used')      
        model = models.vgg16(pretrained=True)
        print('Network loaded')    
    return model
                  
 #Define a new untrained feed-forward network as a classifier , using ReLU activations and dropout  
#output layer must match flower categories =102

def classifier(model, hidden_units) : 
#check that the layers are the same as original
#use input feature for trained network used 
    input = model.classifier[0].in_features
    if hidden_units == None: 
        hidden_units = 4000
    
    classifier = nn.Sequential(OrderedDict([
                         ('fc1', nn.Linear(input, hidden_units)),
                         ('relu', nn.ReLU()),
                         ('dropout', nn.Dropout(p=0.5)),                            
                         ('fc2', nn.Linear(hidden_units, 500)),
                         ('relu2', nn.ReLU()),
                         ('dropout', nn.Dropout(p=0.5)), 
                         ('fc3', nn.Linear(500, 102)),
                         ('output', nn.LogSoftmax(dim=1))]))

    #freeze parameters
    for param in model.parameters():
        param.requires_grad = False           
    #replace classifier
    model.classifier = classifier
    print('classifier defined')                
    return model 



def validation(model, valid_loader, device, criterion):
    print('start validation')
    validation_loss = 0 
    validation_accuracy = 0 
    model.to('cuda') 
    model.eval()
    for images, labels in valid_loader:
        images, labels = images.to('cuda'), labels.to('cuda')
        output = model.forward(images)
        validation_loss += criterion(output, labels).item()

        #track accuracy 
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        validation_accuracy += equality.type(torch.FloatTensor).mean()
    return validation_loss, validation_accuracy
    print('end validation')
    
    
def train_model(epochs, train_loader, valid_loader, device, model, optimizer, criterion):
    print('start train_model')
    #passes through the data set
    epochs = 5
    steps = 0
    print_every = 30
    running_loss = 0
    model.to(device) 

    #start training network
    for e in range(epochs):
        model.train()
        for images, labels in train_loader:
            steps += 1
            images, labels = images.to(device), labels.to(device)
            #Clear the gradients, do this because gradients are accumulated
            optimizer.zero_grad()        
            #forward & backward passes 
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            #update the weights using the optimizer
            optimizer.step() 
            running_loss += loss.item() 
            if steps % print_every == 0:
                #network in eval mode for inference    
                model.eval()    

                # 2/20  --need to turn validation gradient off to save mem.
                with torch.no_grad():
                     validation_loss, validation_accuracy = validation(model, valid_loader, device, criterion)

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Validation Loss: {:.3f}.. ".format(validation_loss/len(valid_loader)),
                      "Validation Accuracy: {:.3f}".format(validation_accuracy/len(valid_loader)))                

                running_loss = 0
                #turn training back on
                model.train()
    print ("Training complete")
    return model 

                  
#  validation on the test set
def test_model(model, test_loader, device, optimizer, criterion):
    print('start test_model')
    test_correct = 0
    test_total = 0
   #if model.eval doesnt work try:float tensor             
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    
    print('Accuracy of the network on the test images: %d %%' % (100 * test_correct / test_total))
    return test_correct, test_total
   # test_model(model)


# Save the checkpoint 
def save_checkpoint(epochs, model, train_data, save_dir, arch, optimizer):
     model.class_to_idx = train_data.class_to_idx
     checkpoint = {'arch': 'vgg16',
                  'classifier': model.classifier,
                  'class_to_idx': model.class_to_idx,
                  'state_dict': model.classifier.state_dict(),
                  'epochs': epochs,
                  'optimizer_state': optimizer.state_dict
                  }
     torch.save(checkpoint, save_dir)
     print('end save checkpoint')

#call main fcn to run the script
if __name__ == '__main__':
     main()     
     
                  
                                 
                                 
                                 
                                 
                                 
                                 
                                 
                                 
                                 