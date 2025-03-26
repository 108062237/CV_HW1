# NYCU CV2025 - Homework 1: Image Classification

**Student ID:** 313551113
**Name:** 王唯誠

## 📌 Task
Classify 100 image categories using models from the ResNet family.  
Predict labels for 2,344 unlabeled test images and submit results to CodaBench.

## 🧠 Models Used
- ResNet101
- ResNeXt50_32x4d
- ResNeXt101_32x8d
- ResNeXt101_32x8d + SE Block  
All models use pretrained weights and a modified final classifier (Dropout + FC Layer).


## 📄 Files

- `train.py`: Main script for training the model  
- `data_loader.py`: Custom PyTorch datasets and data loaders for training and testing  
- `ensemble.py`: Combines predictions from multiple models 
- `evaluate.py`: Computes validation accuracy and loss during training  
- `inference.py`: Runs inference on the test set and outputs `prediction.csv`  
- `utils.py`: Utility functions 
- `model/model.py`: Model architecture definitions 
- `main.py`: Entry point to run training ans evaluation 
- `configs/config`: YAML or Python config files for model and training parameters  

