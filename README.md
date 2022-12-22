# project2_brainAge

## General Information:

This repository contains the code and documentation for the ML4 Science project (project 2 option A) that our team worked on with the Medical Image Processing lab at EPFL, entitled "Predicting Biological Age for Brain Anatomical Volume Measurements".

## Team:

The project was accomplished by: 
- Antonin Faure: @antoninfaure
- Manon Dorster : @mdorster
- Alexandre Maillard : @AlexMlld

## Introduction to the project:

The aim of our project was to train and test various models combined with advanced feature engineering to predict the chronological age of patients based on brain anatomical volume measurements. We worked with the Medical Image Processing lab at EPFL, who provided us with datasets.

# Project structure: 

## Data:

We were provided with two data sets:
- dataset 1 : a set of 335 features for 133 patients aged between 68 and 85 years
- dataset 2 : a set of 175 features for 231 patients aged between 49 and 73 years.
  
**Note**
To run our code, please download these files and put them in the same folder as our code files. 

## Helpers file
The [helpers.py](helpers.py) contains all the useful functions used in our scripts [Data Visualization.ipynb], [Clustering.ipynb], [Optimization.ipynb], and [Models.ipynb].

## Clustering:

We divided the patients in each dataset into subgroups upon the lab's request, to see if there exists different aging patterns among patients. We used the subgroups later for model training and compared to models trained on the entire datasets. The aim was to see if we would get more accurate age predictions by clustering the data points. 

The script [Clustering.ipynb](Clustering.ipynb) outputs statistics on the cluster distributions using GMM and K-means clustering.

## Data visualization:

The script [Data Visualization.ipynb](Data Visualization.ipynb) outputs statistics on the datasets, with and without feature decomposition (raw, PCA, PLS).

## Model training: 

All models (elastic net regression, KNN, random forest, Adaboost, SVR) were trained on both the global dataset (global model) and each cluster (local models), for each type of feature engineering method (original features or raw, lasso regularization, PLS regression, PCA decomposition) with or without correlation filtering (MI score filtering).

Please run the script [Models.ipynb](Models.ipynb) to train all model combinations.

## Model optimization: 

We implemented an algorithm to compare the global and local models and determine which ones yield the most accurate results based on R2 score an mean average error (MAE).
The script [Optimization.ipynb](Optimization.ipynb) produces the optimized results (see report for more details).

## Report:

A 4 page scientific report describes the entirety of the project, the goals that we were aiming for, and the workflow we established to obtain our results. The optimal prediction results we obtained are all clearly displayed in the appendix of the report.
