# AD-BrainAge-Asymmetry
# Hemispheric Asymmetry Shapes Brain Age Gap to Predict Cognitive Impairment Across the Alzheimer’s Disease Spectrum

This repository provides the analysis scripts to reproduce the brain age prediction (BAG) and hemispheric structural asymmetry (AM) computation methods proposed in the study, as well as the subsequent modeling and classification of the Alzheimer’s disease continuum (CN, SCD, MCI, AD) based on their integration. It covers the complete experimental pipeline, from MRI data loading and deep learning model training to downstream machine learning–based evaluation.

# Abstract
Alzheimer’ s disease (AD), the most common cause of dementia, remains incurable, highlighting the need for biomarkers that detect early-stage pathology with high sensitivity. Here we present an imaging framework that integrates brain age gap (BAG)—a marker of accelerated aging—with regional asymmetry metrics (AM) to improve classification across the AD spectrum. Using structural MRI from 13,913 cognitively normal individuals, we trained a compact deep learning model to estimate brain age and applied it to independent cohorts including cognitively normal (CN), subjective cognitive decline (SCD), mild cognitive impairment (MCI), and AD. We further quantified hemispheric AM across 34 cortical and 4 subcortical and combined these metrics with BAG to predict cognitive state. BAG increased progressively from CN through SCD and MCI to AD, while AM provided complementary stage-specific sensitivity by capturing lateralized neurodegeneration. Interpretability analyses highlighted asymmetry in the inferior temporal gyrus, amygdala, and precuneus as key correlates of abnormal brain aging. Together, these findings establish BAG–AM as an interpretable imaging biomarker that links structural brain aging with hemispheric asymmetry, offering mechanistic insight and enhancing early detection of cognitive impairment.

# Folder structure
models/                - Contains neural network model definitions  
dataset.py             - Script for loading MRI data and corresponding labels, and constructing datasets  
tbrain_age_train.py    - Training script for deep learning–based brain age (BAG) prediction  
adni_pairwise_roc.py   - Classification of the AD continuum (CN/SCD/MCI/AD) and pairwise ROC analysis  
README.md              - Project documentation and usage instructions

