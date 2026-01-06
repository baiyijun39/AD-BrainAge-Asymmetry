This repository provides the analysis scripts to reproduce the brain age prediction (BAG) and hemispheric structural asymmetry (AM) computation methods proposed in the study, as well as the subsequent modeling and classification of the Alzheimer’s disease continuum (CN, SCD, MCI, AD) based on their integration. It covers the complete experimental pipeline, from MRI data loading and deep learning model training to downstream machine learning–based evaluation.

# Folder structure
models/                - Contains neural network model definitions  
dataset.py             - Script for loading MRI data and corresponding labels, and constructing datasets  
tbrain_age_train.py    - Training script for deep learning–based brain age (BAG) prediction  
adni_pairwise_roc.py   - Classification of the AD continuum (CN/SCD/MCI/AD) and pairwise ROC analysis  
README.md              - Project documentation and usage instructions

