#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#                                                              
# PROGRAMMER: Andres Iglesias
# DATE CREATED: 25/05/2024                                 
# REVISED DATE: 
# PURPOSE: Predict flower name from an image with predict.py along with the probability of
# that name. Pass in a single image /path/to/image and return the flower name and class 
# probability.
# BASIC USAGE: python predict.py /path/to/image checkpoint
# OPTIONS: * Return top K most likely classes:
#            python predict.py input checkpoint --top_k 3 
#          * Use a mapping of categories to real names: 
#            python predict.py input checkpoint --category_names cat_to_name.json 
#          * Use GPU for inference:
#            python predict.py input checkpoint --gpu 
# VERSIONS: torch <2.3.0>
#           torchvision <0.18.0>
##

# Imports python modules
import time
import helper


# Imports functions created for this program
from get_input_args import get_predict_args

# Main program function defined below
def main():
    # Measures total program runtime by collecting start time
    start_time = time.time()
    
    # Retrieving Command Line Arugments from user as input from
    # ruunning the program from a terminal window. This function returns
    # the collection of these command line arguments from the function call as
    # the variable in_arg
    in_arg = get_predict_args()
       
    # Printing set input values
    print(" ")
    print(f"Path to image: {in_arg.path_to_image}")
    print(f"Path to checkpoint: {in_arg.path_to_checkpoint}")
    print(f"Number of top K classes: {in_arg.top_k}")
    print(f"Path to category names file: {in_arg.category_names}")
    print(f"GPU: {in_arg.gpu}\n")
    
    # Function for loading and rebuilding a model from a checkpoint file.
    # This function returns a tuple containing the model rebuilt from checkpoint file and
    # the optimizer object with their state_dict() loaded.
    checkpoint_model, optimizer = helper.load_checkpoint(in_arg.path_to_checkpoint)
    
    # Function for predicting the top_k classes of an image.
    # This function returns the top_k probabilities and its corresponding top classes
    # In two separate lists
    top_ps, top_classes = helper.predict_class(in_arg.path_to_image, checkpoint_model, in_arg.top_k, in_arg.gpu)
    
    # Function for predicting the top_k classes of an image.
    # This function returns the top_k probabilities and its corresponding top classes
    # In two separate lists
    helper.display_class(top_ps, top_classes, in_arg.category_names)
    
    
    
 
    end_time = time.time()
    
    # Computes overall runtime in seconds & prints it in hh:mm:ss format
    tot_time = end_time - start_time #calculate difference between end time and start time
    print("\n** Total Elapsed Runtime:",
          str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
          +str(int((tot_time%3600)%60)))
    

# Call to main function to run the program
if __name__ == "__main__":
    main()