---
license: apache-2.0
language:
- en
metrics:
- accuracy
- f1
- recall
- precision
base_model:
- dslim/distilbert-NER
pipeline_tag: token-classification
---
The Model itself [here](https://huggingface.co/dimanoid12331/distilbert-NER_finetuned_on_mountines).

It is fine-tuned [DistilBERT-NER](https://huggingface.co/dslim/distilbert-NER) model with the classifier replaced to increase the number of classes from 9 to 11. Two additional classes is I-MOU and B-MOU what stands for mountine.
Inital new classifier inherited all weights and biases from original and add new beurons wirh weights initialized wirh xavier_uniform_

#### How to use

This model can be utilized with the Transformers *pipeline* for NER, similar to the BERT models.

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
tokenizer = AutoTokenizer.from_pretrained("dimanoid12331/distilbert-NER_finetuned_on_mountines")
model = AutoModelForTokenClassification.from_pretrained("dimanoid12331/distilbert-NER_finetuned_on_mountines")
nlp = pipeline("ner", model=model, tokenizer=tokenizer)
example = "My name is Wolfgang and I live in Berlin"
ner_results = nlp(example)
print(ner_results)
```

Or you can just download inference_model.py run it and follow instructions

## Training data

This model was fine-tuned on English castom arteficial dataset with sentances wich contains mountains. 

As in the dataset, each token will be classified as one of the following classes:

Abbreviation|Description
-|-
O|Outside of a named entity
B-MISC |Beginning of a miscellaneous entity right after another miscellaneous entity
I-MISC | Miscellaneous entity
B-PER |Beginning of a person’s name right after another person’s name
I-PER |Person’s name
B-ORG |Beginning of an organization right after another organization
I-ORG |organization
B-LOC |Beginning of a location right after another location
I-LOC |Location
B-MOU |Beginning of a Mountain right after another Mountain
I-MOU |Mountain

Sentences |Tokens
-|-
216 |2783


## Eval results
| Metric     | Score |
|------------|-------|
| Loss       | 0.2035|
| Precision  | 0.8536|
| Recall     | 0.7906|
| F1         | 0.7117|
| Accuracy   | 0.7906|
