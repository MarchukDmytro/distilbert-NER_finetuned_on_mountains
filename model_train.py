import torch
from datasets import load_from_disk
from torch.utils.data import random_split
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, Trainer, TrainingArguments

best_dir = None
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

save = input_function(str ,"Are you want to save best epoch model separatly, best please type yes/no: ", "Please type yes or no",['yes','no'])

if save == 'no':
    output_dir = input_function(str,"Please enter a output directory: ",'')
if save == 'yes':
    best_dir = input_function(str,"Please enter a output directory for best save: ",'')
    output_dir = input_function(str,"Please enter a output directory for epoch saves: ",'') 

learning_rate = input_function(float ,"Please enter a learning rate: ", "it must be a number, can be in format xe-n")                
batch_size = input_function(int ,"Please enter a batch size: ", "it must be a number")      
num_train_epochs= input_function(int ,"Please enter a epoch number: ", "it must be a number")                 
dataset = input_function(load_from_disk,"Please enter a path to dataset: ","it must be a valid path")

#load the model and tokenizer
tokenizer = input_function(
    AutoTokenizer.from_pretrained,  # Load a pretrained tokenizer
    "Please enter a path to a model: ",  # Prompt message for user input
    "it must be a valid path"  # Error message if the input is invalid
)

# Load a pretrained model for token classification using the tokenizer's path
model = AutoModelForTokenClassification.from_pretrained(tokenizer.name_or_path)

old_classifier = model.classifier  # Save the old classifier(last layer of the model)

model.classifier = torch.nn.Linear(model.config.hidden_size, 11) # creale new linear layer for icrease output number from 9 to 11

torch.nn.init.xavier_uniform_(model.classifier.weight) # generate new weights

with torch.no_grad():
    model.classifier.weight[:old_classifier.weight.size(0), :] = old_classifier.weight  # Copy old weights
    model.classifier.bias[:old_classifier.bias.size(0)] = old_classifier.bias  # Copy old bias


# changing model metadata to match new one
model.num_labels = 11 

model.config.id2label = {
                        0: 'O', 
                        1: 'B-PER', 
                        2: 'I-PER', 
                        3: 'B-ORG', 
                        4: 'I-ORG', 
                        5: 'B-LOC', 
                        6: 'I-LOC', 
                        7: 'B-MISC', 
                        8: 'I-MISC', 
                        9: 'B-MOU', 
                        10: 'I-MOU'
                        }


def tokenize_and_align_labels(examples): # prepare data for training
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True) #tokenize the words

    labels = []

    for i, label in enumerate(examples["ner_tags"]): # labeling tokens
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # take word ids for sentance
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # Ignore the special tokens in the start and end of sentance
            else:  # Add label for tokens
                label_ids.append(label[word_idx])
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Tokenize dataset
tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

# Create data collator for padding inputs
data_collator = DataCollatorForTokenClassification(tokenizer)

# Set training arguments
training_args = TrainingArguments(
    output_dir=output_dir,  # Output directory for epoch save
    evaluation_strategy="epoch",         # Evaluate every epoch
    learning_rate=learning_rate,                  # Learning rate
    per_device_train_batch_size=batch_size,      # Batch size
    per_device_eval_batch_size=batch_size,       # Batch size for evaluation
    num_train_epochs=num_train_epochs,                  # Number of epochs
    weight_decay=0.01,                   # Strength of weight decay
    load_best_model_at_end=True,
    save_strategy='epoch',
)
#split dataset to train/test subsets
train_size = (int(len(tokenized_datasets)*0.8))
test_size = len(tokenized_datasets)- train_size

train_set,eval_set = random_split(tokenized_datasets,[train_size,test_size])

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_set,
    eval_dataset=eval_set,
    tokenizer=tokenizer,
    data_collator=data_collator, 
)

# Train the model
trainer.train()

#save best output
if best_dir:
    model.save_pretrained(best_dir)
    tokenizer.save_pretrained(best_dir)