# Reflecting-Topology-Consistency-and-Abnormality-via-Learnable-Attentions-for-Airway-Labeling


> By Chenyu Li, Minghui Zhang, Chuyan Zhang, Yun Gu
>> Institute of Medical Robotics, Shanghai Jiao Tong University
>> Department of Automation, Shanghai Jiao Tong University, Shanghai, China


## Dependencies 
- torch, 2.2.2
- python, 3.11.8
- numpy，1.26.4

## Usage
### Data Processing
To generate graph dataset from the airway:
```
python data_process/process_data.py
```
This script performs the following:

Parses the airway based on bifurcation points (each branch is assigned a unique value in the volume).

Computes topological relationship of branches (parent map and children map).

Extract features required for airway labeling.

### Training
To train the model:
```
python train/train.py
```
The training process uses the graph dataset generated from the data processing step.
### Testing
To evaluate the model:
```
python test/test.py
```
The testing script evaluates the performance of the trained model on the test dataset.

## Configuration
All model parameters, paths, and other configurations are stored in the file:
```
config/config.py
```

## Label Mapping
You can refer to the file:
```
config/anno_class_dict.json
```
to understand the mapping between labels and anatomical names. 
This file provides a dictionary that maps each label to its corresponding anatomical name.