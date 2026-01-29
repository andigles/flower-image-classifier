#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# PROGRAMMER: Andres Iglesias
# DATE CREATED: 25/05/2024
# REVISED DATE:
# PURPOSE: Create two functions that retrieves the train.py and
#          predict.py command line inputs from the user using the
#          Argparse Python module. If the user fails to provide some
#          or all of the inputs, then the default values are used for
#          the missing inputs.

#
##

# Import python modules
import argparse


def get_train_args():
    """
    Create a function that retrieves the following command line inputs
    from the user using the Argparse Python module. If the user fails to
    provide some or all of the inputs, then the default values are
    used for the missing inputs. Command Line Arguments:
     1. Working directory path with the image data as data_dir
     2. Directory to save checkpoints as --save_dir with default value 'save_directory'
     3. Pretrained Model Architecture as --arch with default value 'resnet50'
     4. Learing rate hyperparameter as --learning_rate with default value 0.0005
     5. Number of hidden units for the classifier as --hidden_units. It accepts multiple
        layers e.g. --hidden_units 512 256 64. Default value 512
     6. Dropout value for overfitting as --dropout with default value 0.3
     7. Number of epochs used for training as --epochs with default value 5
     8. Use GPU for training if available as --gpu. Default use CPU. If not available
        will run in CPU
    """
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()

    # Create command line arguments as mentioned above using add_argument() from ArguementParser method
    parser.add_argument("data_dir", help="The data folder")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="save_directory/",
        help="path to the folder used for saving checkpoints",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="resnet50",
        help="The pretrained model architecture to use",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0005,
        help="Learning rate value used for training",
    )
    parser.add_argument(
        "--hidden_units",
        nargs="*",
        type=int,
        default=512,
        help="Hidden units e.g. --hidden_units 512 256 64",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.2, help="Dropout value for overfitting"
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of epochs used for training"
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="If activated, run training in GPU mode if available",
    )

    # Replace None with parser.parse_args() parsed argument collection that
    # you created with this function
    return parser.parse_args()


def get_predict_args():
    """
    Create a function that retrieves the following command line inputs
    from the user using the Argparse Python module. If the user fails to
    provide some or all of the inputs, then the default values are
    used for the missing inputs. Command Line Arguments:
     1. Path to the image file as path_to_image
     2. Path to the checkpoint file as path_to_checkpoint
     3. Top K most likely classes as --top_k with default value 1
     4. Use a mapping to categorical name in path to file as --category_names with default
     value 'cat_to_name.json'
     5. Use GPU for inference if available as --gpu. Default use CPU. If not available
        will run in CPU
    """
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()

    # Create command line arguments as mentioned above using add_argument() from ArguementParser method
    parser.add_argument("path_to_image", help="The image path")
    parser.add_argument("path_to_checkpoint", help="The checkpoint path")
    parser.add_argument(
        "--top_k", type=int, default=1, help="Int number for k most likely classes"
    )
    parser.add_argument(
        "--category_names",
        type=str,
        default="cat_to_name.json",
        help="Mapping to categorical name in path to file",
    )

    parser.add_argument(
        "--gpu",
        action="store_true",
        help="If activated, run training in GPU mode if available",
    )

    # Replace None with parser.parse_args() parsed argument collection that
    # you created with this function
    return parser.parse_args()
