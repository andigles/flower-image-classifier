#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# PROGRAMMER: Andres Iglesias
# DATE CREATED: 25/05/2024
# REVISED DATE:
# PURPOSE: Helper file which handles all the functions necesary for
#          running train.py and predict.py scripts.
# VERSIONS: torch <2.3.0>
#           torchvision <0.18.0>
##

# Imports python modules
import torch
import numpy as np
import time
import os
import json
import sys
from torch import nn, optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from collections import OrderedDict
from PIL import Image

try:
    # torchvision >= 0.13
    from torchvision.models import get_model, get_model_weights
except ImportError:
    get_model = None
    get_model_weights = None

def error_exit(message, code=2):
    """Print a freindly error and exit with a non-zero code (no stack trace)."""
    print(f"Error: {message}", file=sys.stderr)
    raise SystemExit(code)

def validate_file(path, label):
    if not path or not os.path.isfile(path):
        error_exit(f"{label} not found: {path}")

def validate_dir(path, label):
    if not path or not os.path.isdir(path):
        error_exit(f"{label} not found: {path}")

def validate_top_k(top_k, max_k = 102):
    if not isinstance(top_k, int):
        error_exit("--top_k must be an integer")
    if top_k <= 0:
        error_exit("--top_k must be a positive integer")
    if top_k > max_k:
        error_exit(f"--top_k must be <= {max_k}")

def data_batching(data_dir):
    """
    Function for getting loaders for train, validation and testing data from data_dir.

    This function performs the following steps:
    1. Defines the directories for the training, validation, and testing datasets.
    2. Training data augmentation and Data normalization
    3. Data loading
    4. Data batching

    Parameters:
    data_dir (str): The directory containing the 'train', 'valid', and 'test' directories.

    Returns:
    tuple: A tuple containing the DataLoader objects for the training, validation, and testing datasets.
    class_to_idx (dict): Mapping of classes to indices

    """

    train_dir = data_dir + "/train"
    valid_dir = data_dir + "/valid"
    test_dir = data_dir + "/test"

    # Training data augmentation and Data normalization
    train_transforms = transforms.Compose(
        [
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    valid_transforms = transforms.Compose(
        [
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    test_transforms = transforms.Compose(
        [
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # Data loading
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    # Data batching
    trainloader = DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = DataLoader(valid_data, batch_size=64)
    testloader = DataLoader(test_data, batch_size=64)

    # Mapping of classes to indices
    class_to_idx = train_data.class_to_idx

    return trainloader, validloader, testloader, class_to_idx

def building_model(model_arch, hidden_units, dropout):
    """
    Function for building the complete model based on a pretrained model and a
    custom classifier based on hyperparameters specified from input.

    Parameters:
    model_arch (str): Model architecture which gonna be used for pretrained model.
    hidden_units (list): List with number of units of each hidden layer
    dropout (float): Dropout value to help with overfitting

    Returns:
    model (class 'torchvision.models): Complete model network with pretrained model and
    classifier according to hyperparameters provided.
    """

    # Pretrained Network
    if get_model is not None and get_model_weights is not None:
        # Use the modern weights API (no deprecation warnings)
        weights_enum = get_model_weights(model_arch)
        weights = weights_enum.DEFAULT  # pretrained weights
        model = get_model(model_arch, weights=weights)
    else:
        # Fallback for older torchvision (may warn)
        model = getattr(models, model_arch)(pretrained=True)

    # Freeze parameters in pre-trained model
    for param in model.parameters():
        param.requires_grad = False

    # Get the last layer name and in_features values
    last_layer_name = list(model._modules.keys())[-1]
    last_layer = getattr(model, last_layer_name)

    if isinstance(last_layer, nn.Sequential):
        first_linear = [
            module for module in last_layer.modules() if isinstance(module, nn.Linear)
        ][0]
        last_layer_in_features = first_linear.in_features
    else:
        last_layer_in_features = last_layer.in_features

    # Create the classifier
    classifier = create_classifier(last_layer_in_features, 102, hidden_units, dropout)

    # Replace last layer with our classifier
    model._modules[last_layer_name] = classifier

    return model


def create_classifier(input_size, output_size, hidden_units, dropout):
    """
    Function for creating classifier.

    Parameters:
    input size (int): Number of inputs for the classifier.
    output size (int): Number of outputs for the classifier.
    hidden_units (list): List with number of units of each hidden layer for the classifier
    dropout (float): Dropout value to help with overfitting for the classifier

    Returns:
    classifier (torchvision.models): classifier network layer built from
    the hyperparameters provided.
    """
    # Add the first layer, input to a hidden layer
    hidden_units = [input_size] + hidden_units
    layers = OrderedDict()

    for i in range(len(hidden_units) - 1):
        # Add a fully connected layer with ReLU activation and dropout
        layers["fc{}".format(i + 1)] = nn.Linear(hidden_units[i], hidden_units[i + 1])
        layers["relu{}".format(i + 1)] = nn.ReLU()
        layers["dropout{}".format(i + 1)] = nn.Dropout(p=dropout)

    # Add the output layer
    layers["output"] = nn.Linear(hidden_units[-1], output_size)
    layers["softmax"] = nn.LogSoftmax(dim=1)

    # Create the classifier
    classifier = nn.Sequential(layers)

    return classifier


def training_classifier(
    model, trainloader, validloader, testloader, learning_rate, epochs, use_gpu
):
    """
    Function for training classifier of the model. The parameters of the pretrained
    model are freezed while training the layers for the classifier created.

    Parameters:
    model (torchvision.models): Complete model built from pretrained model and classifier
    trainloader (torch.utils.data.DataLoader): Dataloader batching training data.
    validloader (torch.utils.data.DataLoader): Dataloader batching validation data.
    testloader (torch.utils.data.DataLoader): Dataloader batching test data.
    learning_rate (float): Learning rate value used for training.
    epochs (int): Number of epochs used for training
    use_gpu (bool): If activated, run training in GPU mode if available

    Returns:
    model (torchvision.models): Complete model network with pretrained model and
                                classifier according to hyperparameters provided.
    optimizer.state_dict()(dict): State dictionary for optimizer used in training
    train_losses (list): List containing train loss of each epoch
    valid_losses (list): List containing validation loss of each epoch
    """
    # Use GPU if available
    if use_gpu:
        # If CUDA is avaiable run
        print("Trying GPU.. ")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if device == torch.device("cpu"):
            print("GPU is not availabe")
        else:
            print("Running in GPU.")
    else:
        device = "cpu"

    # Model to CUDA if available
    model.to(device)

    # Define criterion, optimizer, epochs and other variables for classifier training
    criterion = nn.NLLLoss()

    last_layer_name = list(model._modules.keys())[-1]
    classifier = getattr(model, last_layer_name)

    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)

    print_every = 20

    for e in range(epochs):
        # Training loop
        running_train_loss = 0
        t_start_time = time.time()
        no_batches = 0

        for inputs, labels in trainloader:
            if no_batches % print_every == 0:
                print(f"Training in {device}.. ")

            model.train()
            no_batches += 1

            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            running_train_loss += loss.item()

            loss.backward()
            optimizer.step()

            if no_batches % print_every == 0 or no_batches == len(trainloader):
                # Validation Loss and Accuracy
                print(f"Validating in {device}.. ")
                running_valid_loss = 0
                accuracy = 0
                v_start_time = time.time()

                with torch.no_grad():
                    # Validation loop
                    model.eval()
                    for inputs, labels in validloader:
                        # Move input and label tensors to the default device
                        inputs, labels = inputs.to(device), labels.to(device)

                        logps = model.forward(inputs)
                        loss = criterion(logps, labels)
                        running_valid_loss += loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                # For validating time
                v_end_time = time.time()
                v_tot_time = (
                    v_end_time - v_start_time
                )  # calculate difference between end time and start time
                print(
                    "Validating time:",
                    str(int((v_tot_time / 3600)))
                    + ":"
                    + str(int((v_tot_time % 3600) / 60))
                    + ":"
                    + str(int((v_tot_time % 3600) % 60))
                    + "\n",
                )

                train_loss = running_train_loss / no_batches
                valid_loss = running_valid_loss / len(validloader)

                print(
                    f"Epoch {e + 1}/{epochs}.. "
                    f"Train loss: {train_loss:.3f}.. "
                    f"Validation loss: {valid_loss:.3f}.. "
                    f"Validation accuracy: {accuracy / len(validloader):.3f}"
                )
                print("--------------------------------------------------")

                running_valid_loss = 0

        # For training time
        t_end_time = time.time()
        t_tot_time = (
            t_end_time - t_start_time
        )  # calculate difference between end time and start time
        print(
            "Epoch time:",
            str(int((t_tot_time / 3600)))
            + ":"
            + str(int((t_tot_time % 3600) / 60))
            + ":"
            + str(int((t_tot_time % 3600) % 60))
            + "\n",
        )

    return model, optimizer.state_dict()


def saving_model(
    save_dir,
    class_to_idx,
    trained_model,
    optimizer_state_dict,
    arch,
    learning_rate,
    hidden_units,
    dropout,
    epochs,
):
    """
    Function for saving trained model and hyperparameters in spedified save_dir folder.
    It creates a new checkpint file everytime is used under the name checkpoint_{i}.pth.
    in the specified folder.

    Parameters:
    save_dir (str): Folder to save checkpoint file
    class_to_idx (dict): Mapping of classes to indices
    trained_model (torchvision.models): Complete trained model object
    optimizer_state_dict (dict): State dict for optimizer
    arch (str): The pretrained model architecture to use
    learning_rate (float): Learning rate used while training
    hidden_units (list): List with number of units of each hidden layer
    dropout (float): Dropout value to help with overfitting
    epochs (int): Number of epochs used for training

    Returns:
    None
    """

    # Saving the model
    checkpoint = {
        "class_to_idx": class_to_idx,
        "state_dict": trained_model.state_dict(),
        "optimizer_state_dict": optimizer_state_dict,
        "architecture": arch,
        "learning_rate": learning_rate,
        "hidden_units": hidden_units,
        "dropout": dropout,
        "epochs": epochs,
        "output_size": 102,
    }
    
    base_filename = "checkpoint"
    extension = ".pth"
    i = 0
    filename = base_filename + extension

    # Check if the file exists, if so, create a new filename
    while os.path.exists(os.path.join(save_dir, filename)):
        i += 1
        filename = f"{base_filename}_{i}{extension}"

    os.makedirs(save_dir, exist_ok=True)
    torch.save(checkpoint, os.path.join(save_dir, filename))

    return None


def load_checkpoint(path_to_checkpoint):
    """
    Function for loading and rebuilding a model from a checkpoint file.

    Parameters:
    path_to_checkpoint (str): Path to the checkpoint file

    Returns:
    checkpoint_model (torchvision.models): Complete model network with pretrained model and
                                           classifier according checkpoint file.
    optimizer (torch.optim.adam.Adam): Optimizer object as per trained in checkpoint
    """
    # Load checkpoint
    try:
        checkpoint = torch.load(
            path_to_checkpoint,
            map_location=lambda storage, loc: storage,
            weights_only=False,
        )
    except TypeError:
        # Older PyTorch: no weights_only argument
        checkpoint = torch.load(
            path_to_checkpoint,
            map_location=lambda storage, loc: storage,
        )

    arch = checkpoint["architecture"]

    if get_model is not None and get_model_weights is not None:
        weights_enum = get_model_weights(arch)
        weights = weights_enum.DEFAULT
        checkpoint_model = get_model(arch, weights=weights)
    else:
        checkpoint_model = getattr(models, arch)(pretrained=True)

    # Freeze parameters in pre-trained model
    for param in checkpoint_model.parameters():
        param.requires_grad = False

    # Get the last layer name and in_features values
    last_layer_name = list(checkpoint_model._modules.keys())[-1]
    last_layer = getattr(checkpoint_model, last_layer_name)

    if isinstance(last_layer, nn.Sequential):
        first_linear = [m for m in last_layer.modules() if isinstance(m, nn.Linear)][0]
        last_layer_in_features = first_linear.in_features
    else:
        last_layer_in_features = last_layer.in_features

    # Backward compatible:
    # - Old checkpoints store the classifier object
    # - New checkpoints rebuild classifier from hyperparameters
    if "classifier" in checkpoint:
        checkpoint_model._modules[last_layer_name] = checkpoint["classifier"]
    else:
        output_size = checkpoint.get("output_size", 102)
        classifier = create_classifier(
            last_layer_in_features,
            output_size,
            checkpoint["hidden_units"],
            checkpoint["dropout"],
        )
        checkpoint_model._modules[last_layer_name] = classifier

    classifier = getattr(checkpoint_model, last_layer_name)

    checkpoint_model.class_to_idx = checkpoint["class_to_idx"]
    checkpoint_model.load_state_dict(checkpoint["state_dict"])
    optimizer = optim.Adam(classifier.parameters(), lr=checkpoint["learning_rate"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint_model, optimizer


def predict_class(path_to_image, checkpoint_model, top_k, use_gpu):
    """Predict the class (or classes) of an image using a trained deep learning model."""
    im = process_image(path_to_image)
    inputs = torch.from_numpy(np.array([im])).float()

    # Use GPU if available
    if use_gpu:
        # If CUDA is avaiable run
        print("Trying GPU.. ")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if device == torch.device("cpu"):
            print("GPU is not availabe")
        else:
            print("Running in GPU.")
    else:
        device = "cpu"

    inputs = inputs.to(device)
    checkpoint_model.to(device)

    with torch.no_grad():
        checkpoint_model.eval()
        logps = checkpoint_model.forward(inputs)
        ps = torch.exp(logps)
        ps = ps.topk(top_k, dim=1)
        top_indices = np.array(ps[1][0].cpu().numpy())
        idx_to_class = {x: y for y, x in checkpoint_model.class_to_idx.items()}
        top_classes = [idx_to_class[x] for x in top_indices]
        top_p = ps[0].cpu().numpy().tolist()

    return top_p[0], top_classes


def process_image(path_to_image):
    """Scales, crops, and normalizes a PIL image for a PyTorch model,
    returns an Numpy array
    """
    with Image.open(path_to_image).convert("RGB") as im:
        # Resize with shortest side 256 (assuming format is supported)
        # im.thumbnail((256, 256))
        im_width, im_height = im.size

        # Calculate the new size based on the aspect ratio and target shortest side
        if im_width >= im_height:
            # Width is the shortest side
            new_height = 256
            new_width = int(im_width * (new_height / im_height))
        else:
            # Height is the shortest side
            new_width = 256
            new_height = int(im_height * (new_width / im_width))

        im = im.resize((new_width, new_height), Image.LANCZOS)

        # Crop out the center 224 x 224 portion of the image
        im_width, im_height = im.size
        x = (im_width - 224) // 2
        y = (im_height - 224) // 2
        im = im.crop((x, y, x + 224, y + 224))

        # Converting from int 0-255 to floats 0-1 and normalization
        np_im = np.array(im)
        np_im = np_im / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        np_im = (np_im - mean) / std

        # Transpose for color channel to be in first dimension
        np_im = np.transpose(np_im, (2, 0, 1))

        return np_im


def display_class(top_ps, top_classes, category_names):
    """Print category name if available. If not, print the label values"""

    if category_names is not None:
        with open(category_names, "r") as f:
            cat_to_name = json.load(f)

        top_categories = [cat_to_name[str(label)] for label in top_classes]

        if len(top_ps) == 1:
            print(f"Prediction (name): {top_categories[0]}")
            print(f"Probability: {top_ps[0]}")

        else:
            print(f"Prediction (name): {top_categories[0]}")
            print(f"Probability: {top_ps[0]}\n")
            print(f"Top classes (names): {top_categories}")
            print(f"Top probabilites: {top_ps}")

    else:
        if len(top_ps) == 1:
            print(f"Prediction (lable): {top_classes[0]}")
            print(f"Probability: {top_ps[0]}")

        else:
            print(f"Prediction (lable): {top_classes[0]}")
            print(f"Probability: {top_ps[0]}\n")
            print(f"Top classes (labels): {top_classes}")
            print(f"Top probabilites: {top_ps}")
