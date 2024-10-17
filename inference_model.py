import os
import json
import torch
import numpy as np
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def input_function(v_type,text,error_text, if_in = None): #function for user input
    var = input(text)
    try:
        var = v_type(var)
        if not if_in or var in if_in:
            return var
        else:
            print(error_text)
            input_function(v_type,text,error_text,if_in)
    except Exception as e:
        print(e)
        print(error_text)
        return input_function(v_type,text,error_text,if_in)

def convert_floats(obj):
    if isinstance(obj, np.float32):  # Convert float32 to float
        return float(obj)
    if isinstance(obj, np.ndarray):  # Handle NumPy arrays
        return obj.tolist()
    return obj


# Function to obtain user input with validation (input_function is assumed to be defined elsewhere)
tokenizer = input_function(
    AutoTokenizer.from_pretrained,  # Load a pretrained tokenizer
    "Please enter a path to a model: ",  # Prompt message for user input
    "it must be a valid path"  # Error message if the input is invalid
)

# Load a pretrained model for token classification using the tokenizer's path
model = AutoModelForTokenClassification.from_pretrained(tokenizer.name_or_path)


# Prompt user for the output directory where results will be saved
output_dir = input("Please enter a output directory: ")

# Create a pipeline for Named Entity Recognition (NER) using the specified model and tokenizer
pipe = pipeline(task='ner', model=model, tokenizer=tokenizer, device=device)

# If the user did not provide an output directory, use the current working directory
if not output_dir:
    output_dir = os.getcwd()

# Get the input data from the user for NER processing
data = input("Please enter data: ")

# Process the input data through the NER pipeline
output = pipe(data)

# Generate a timestamp to create a unique filename
current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
filename = "output_" + current_time+'.json'  # Create a filename using the current timestamp

# Save the NER output to a JSON file
with open(filename, 'w') as json_file:
    json.dump(output, json_file, default=convert_floats, indent=4)  # Dump the output to JSON format