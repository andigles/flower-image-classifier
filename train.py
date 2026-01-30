#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# PROGRAMMER: Andres Iglesias
# DATE CREATED: 25/05/2024
# REVISED DATE:
# PURPOSE: Train a new network on a data set with train.py. Prints out training
#          loss, validation loss, and validation accuracy as the network trains
# BASIC USAGE: python train.py data_directory
# OPTIONS: * Set directory to save checkpoints:
#            python train.py data_dir --save_dir save_directory
#          * Choose architecture:
#            python train.py data_dir --arch "vgg13"
#          * Set hyperparameters:
#            python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
#          * Use GPU for training: python train.py data_dir --gpu
# VERSIONS: torch <2.3.0>
#           torchvision <0.18.0>
##

# Imports python modules
import time
import helper


# Imports functions created for this program
from get_input_args import get_train_args


# Main program function defined below
def main():
    # Measures total program runtime by collecting start time
    start_time = time.time()

    # Retrieving Command Line Arugments from user as input from
    # ruunning the program from a terminal window. This function returns
    # the collection of these command line arguments from the function call as
    # the variable in_arg
    in_arg = get_train_args()

    # Printing set input values
    print(" ")
    print(f"Data directory: {in_arg.data_dir}")
    print(f"Checkpoint directory: {in_arg.save_dir}")
    print(f"Pretrained model architecture: {in_arg.arch}")
    print(f"Learning rate: {in_arg.learning_rate}")
    print(f"hidden_units: {in_arg.hidden_units}")
    print(f"Dropout: {in_arg.dropout}")
    print(f"Epochs: {in_arg.epochs}")
    print(f"GPU: {in_arg.gpu}\n")

    # Function for getting loaders for train, validation and testing data from data_dir.
    # This function returns a tuple containing the DataLoader objects for the training,
    # validation, and testing datasets.
    trainloader, validloader, testloader, class_to_idx = helper.data_batching(
        in_arg.data_dir
    )

    # Function for getting the complete model, the pretrained and new classifier.
    # This function returns the models object with the structure of the pretrained
    # model and the created classifier at the end.
    model = helper.building_model(in_arg.arch, in_arg.hidden_units, in_arg.dropout)

    # Function for getting the trained model.
    # This function returns the models object after being trained, the optimizer state
    # dictionary, a list of the training losses and a list of validation losses after
    # epoch. Only the created classifier is trained.
    trained_model, optimizer_state_dict = helper.training_classifier(
        model,
        trainloader,
        validloader,
        testloader,
        in_arg.learning_rate,
        in_arg.epochs,
        in_arg.gpu,
    )

    # Function for saving trained model and hyperparameters in spedified save_dir folder.
    # This functions create a file in save_dir folder saving all the model parameters and
    # hyperparameters in a checkpoint file.
    helper.saving_model(
        in_arg.save_dir,
        class_to_idx,
        trained_model,
        optimizer_state_dict,
        in_arg.arch,
        in_arg.learning_rate,
        in_arg.hidden_units,
        in_arg.dropout,
        in_arg.epochs,
    )

    # Measure total program runtime by collecting end time
    end_time = time.time()

    # Computes overall runtime in seconds & prints it in hh:mm:ss format
    tot_time = (
        end_time - start_time
    )  # calculate difference between end time and start time
    print(
        "\n** Total Elapsed Runtime:",
        str(int((tot_time / 3600)))
        + ":"
        + str(int((tot_time % 3600) / 60))
        + ":"
        + str(int((tot_time % 3600) % 60)),
    )


# Call to main function to run the program
if __name__ == "__main__":
    main()
