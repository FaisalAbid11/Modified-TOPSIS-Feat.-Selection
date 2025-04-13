# Modified-TOPSIS-Feat.-Selection
Implementing a modified version of TOPSIS for feature selection and comparing it with other feature selection methods to evaluate its impact on model performance (RFE)

## Overview

This project implements five different **feature selection methods** and compares their performance in feature ranking using a **Random Forest (RF)** model. The methods are:

- **Modified TOPSIS**: Our custom version of the **TOPSIS** (Technique for Order of Preference by Similarity to Ideal Solution) method, modified with robust distance metrics like **Manhattan**, **Euclidean**, and **Chebyshev** for better feature ranking.
- **RFE (Recursive Feature Elimination)**: A method that recursively removes features and builds a model on those remaining to identify the most important features.
- **SelectKBest**: A method that selects the top k features based on statistical tests (e.g., **chi-squared** test).
- **Information Gain**: A measure based on how much information is gained by using a feature to predict the target variable.
- **Normal TOPSIS**: The traditional **TOPSIS** method without the robust distance adjustments.

### Modified TOPSIS:
The **Modified TOPSIS** is our proposed feature selection method where we calculate the distance between features and ideal/worst-case solutions using **robust distance metrics**. The method involves normalizing the dataset, calculating weighted values, and evaluating each feature's importance based on distance from ideal solutions.

### Repository Structure:
```text
your-repository/ │ ├── datasets/ # Folder for storing datasets │ ├── main_dataset_1.csv # Example of a main dataset │ ├── main_dataset_2.csv # Example of another main dataset │ └── main_dataset_3.csv # Another main dataset │ ├── notebooks/ # Folder for Jupyter Notebooks │ ├── feature_selection_topsis.ipynb │ ├── feature_selection_rfe.ipynb │ ├── feature_selection_skb.ipynb │ ├── feature_selection_info_gain.ipynb │ └── model_training.ipynb # The notebook that generates data │ ├── README.md # Main documentation file └── LICENSE # License file (e.g., MIT License)
```

- **`datasets/`**: This folder contains the main datasets required for feature selection and model training.
- **`notebooks/`**: This folder contains the Jupyter Notebooks for different feature selection methods and the model training process.
- **`README.md`**: This file provides an overview and instructions for using the repository.
- **`LICENSE`**: The license under which the repository is shared.

  - 
### Result Highlights:
- **Modified TOPSIS** outperforms all other feature selection methods in terms of model performance across different feature set sizes (**5**, **10**, **15**).
- The results demonstrate that selecting the optimal number of features (k=5, 10, 15) and using an effective feature selection method can significantly enhance model accuracy.
