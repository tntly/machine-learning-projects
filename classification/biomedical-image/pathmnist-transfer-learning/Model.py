import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision
from torchvision import datasets, models, transforms
from tqdm import tqdm

def Set_parameter_requires_grad(model, feature_extracting):
    '''
        Function for switching model parameters gradiant on and off.
        Parameters
        ----------
        model : object of ResNet-18 model initialized from Model.py
        feature_extracting : boolian if true then only last two layers are trained.
    '''
    
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def Initialize_Resnet18_model(num_classes, feature_extract = True, use_pretrained=True):
    '''
        Function for Initializing a ResNet-18 model specific for a certain number of classes.
        Parameters
        ----------
        num_classes : int represents the number of classes of a dataset used for training.
        feature_extract : boolian if true then only last two layers are trained.
        use_pretrained : boolian if true then use weights of a ResNet-18 trained on ImageNet-1000 otherwise randomly initialize weights.
        
        Returns
        ----------
        model_ft : Initialized ResNet-18 model.
    '''
    # Step 1) Obtain the pre-trained model
    # Step 2) Create a base model
    model_ft = models.resnet18(pretrained=use_pretrained)
    # Step 3) Freeze Layers
    Set_parameter_requires_grad(model_ft, feature_extract)
    # Step 4) Add new trainable layers to fit new dataset classes
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    return model_ft

def Create_Model_Optimizer_Criterion(n_classes, feature_extract = True, use_pretrained = True):
    '''
        Function for creating a ResNet-18 model, an Optimizer, and a Loss Criterion. 
        Parameters
        ----------
        n_classes : int represents the number of classes of a dataset used for training.
        feature_extract : boolian if true then only last two layers are trained.
        use_pretrained : boolian if true then use weights of a ResNet-18 trained on ImageNet-1000 otherwise randomly initialize weights.
        
        Returns
        ----------
        model : Initialized ResNet-18 model.
        optimizer : an object of torch.optim that is used for training torch model.
        criterion : an object of torch.nn that is used to represent a loss function used for training torch model.
    '''
    
    model = Initialize_Resnet18_model(n_classes, feature_extract, use_pretrained)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    ## set GPU device    
    # Send the model to GPU
    model = model.to(device)    ## load to GPU

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model.parameters()
    print("Params to learn:")
    if feature_extract:
        # reset params_to_update List
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                # set params_to_update List to params set to be learned
                params_to_update.append(param)
                print("\t" ,name)
    else:
        # everything is to be learned
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                print("\t", name)
                
    optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    return (model, optimizer, criterion)
