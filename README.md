
####  Nested NER 

We take ACE2005 as an example for *NESTED NER* to illustrate the process of data prepration.  

Source files for SEBI-MRC-NER contains a list of json in the format : 


```

```json 
{
"context": "begala dr . palmisano , again , thanks for staying with us through the break .",
"end_position": [
    2,
    4,
    12
    ],
"entity_label": "PER",
"impossible": false,
"qas_id": "4.3",
"query": "3",
"span_position": [
    "1;2",
    "1;4",
    "11;12"],
"start_position": [
    1,
    1,
    11]
}
```


## Descriptions of Directories 

Name | Descriptions 
----------- | ------------- 
log | A collection of training logs in experments.   
script |  Shell files help to reproduce our results.  
data_preprocess | Files to generate MRC-NER train/dev/test datasets. 
metric | Evaluation metrics for Flat/Nested NER. 
model | An implementation of MRC-NER based on Pytorch.
layer | Components for building MRC-NER model. 
data_loader | Funcs for loading MRC-style datasets.  
run | Train / Evaluate MRC-NER models.
config | Config files for BERT models. 


# SEBI-MRC-NER
An automated system to identify pertinent legal objectives, actors and objects from SEBI regulatory documents.​