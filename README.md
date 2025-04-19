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
your-repository/
│
├── datasets/                  
│   ├── feature selection new.csv    
├── src/                  
│   ├── modified_topsis.ipynb
│   └── readme.md
│     
├── README.md                  
└── LICENSE                    
```

- **datasets**: This folder contains the main datasets required for feature selection and model training.
- **src**: This folder contains the Jupyter Notebook for different feature selection methods and the model training process as well as a readme file to explain the modified topsis process.
- **README.md**: This file provides an overview and instructions for using the repository.
- **LICENSE**: The license under which the repository is shared.

##  Installation & How to Run

###  Option 1: Run on Google Colab
**1. You can run the notebooks directly in your browser using Google Colab without any installation.**
   
**Links for Each Notebook :** Click the links provided for each notebook to open it in colab:
- [Modified TOPSIS FS vs Traditional FS](https://colab.research.google.com/github/FaisalAbid11/Modified-TOPSIS-Feat.-Selection/blob/f3c230186e5c0d762ea226f412cff3faabad3dfa/src/modified_topsis.ipynb)

**Open from Colab:**
- Go to [Google Colab](https://colab.research.google.com/).
- Click on **"GitHub"** in the pop-up window.
- Paste the repository URL.
- Press Enter or click the search icon.
- Select the desired notebook (e.g., `modified_topsis.ipynb`).
- It will launch in Colab, and you can run the cells directly in the browser.
  
***Note:You must be signed in with your Google account to use Google Colab.***
  
**2. Upload the required datasets when prompted, or mount your Google Drive.**

- **If you're loading datasets stored in your ***GitHub repository***, use the `raw` file URL :**
Click on the dataset file in the repository. Then, click the "Raw" button located at the top-right corner of the file preview. This will open the raw data in a new tab. From there, copy the URL from the address bar and paste it into the variable that stores the file path. For example: 

```python
import pandas as pd
url = "https://raw.githubusercontent.com/FaisalAbid11/Modified-TOPSIS-Feat.-Selection/refs/heads/main/datasets/feature%20slection%20new.csv"
df = pd.read_csv(url)
```
***Note: Change the URL name with the raw url of required dataset***

- **if you're uploading from your local system, first download it from datasets folder use the code to chose from local system:**
```python
from google.colab import files
uploaded = files.upload()
```
- **For larger or multiple files, it's better to upload file in google drive and mount it:**
```python
from google.colab import drive
drive.mount('/content/drive')
```
***Reminder: If you're not using Google Drive, you'll need to re-upload your files every time you reconnect to a Colab session***

**3. You can run all cell together by selecting Run all from Runtime option in menu bar or by pressing Ctrl+f9. But for better undersating run the notebook cells in order by pressing Ctrl+Enter for selected cell or by clicking the triangular icon/button on top left corner of the cell.**

### Option 2: Run Locally

#### 1. Clone the Repository
```bash
git clone https://github.com/FaisalAbid11/Modified-TOPSIS-Feat.-Selection.git
cd your-repository
```
#### 2. Create a Virtual Environment (Optional but Recommended)
```bash
python -m venv env
source env/bin/activate    
```
#### 3. Install Required Packages
```bash
pip install numpy pandas scikit-learn matplotlib seaborn imblearn
```
#### 4. Run the Jupyter Notebooks
```bash
jupyter notebook
```
Then open the desired notebook (e.g., modified_topsis.ipynb) in your browser.

#### Including Datasets

- The dataset files (e.g., Primary dataset from Indonesian Private University.csv etc.) inside the datasets/ folder and so check and use correct path when using it.

## Complete Process for Feature Selection and Model Training

### **Prepare the Dataset**
Before applying the feature selection methods, we first prepared the dataset:

- **Convert Categorical to Numerical Data**: We encoded categorical features into numerical values for better model compatibility. This is done using techniques like **Label Encoding** or **Ordinal Encoding**.

- **Handle Imbalanced Classes**: **SMOTEEN** (Synthetic Minority Over-sampling Technique) was used for balancing classes due to high imbalance in dataset.


### **Feature Selection**
We used multiple feature selection methods, each with a specific **scoring** or **ranking** technique to evaluate and rank the features.

#### Methods Used:
1. **Modified TOPSIS**: Calculate a score for each feature based on its contribution to the target variable using a custom version of the **TOPSIS** method.
2. **Standard TOPSIS**: Similar to Modified TOPSIS, but without the custom modifications. It ranks features based on their distance from an ideal solution using Euclidean or other distance measures.
3. **RFE (Recursive Feature Elimination)**: RFE ranks features by recursively removing the least important ones and evaluating model performance.
4. **SelectKBest**: We rank features based on a scoring function (e.g., ANOVA F-value).
5. **Information Gain**: We evaluate features based on their entropy and information gain.

Each method ranks the features, and we found the top-ranked features. For evaluation, we used the top **5**, **10**, and **15** features for each method.


### **Data Splitting**
After feature selection, the dataset was splitted into **training** and **testing** sets, using a **70/30 split** ratio.

### **Feature Scaling**
We scaled the features to standardize the data, making it more suitable for models that rely on distance or gradients.


### **Model Training & Hyperparameter Tuning**
Model traing was done using **Random Forest** and performed **GridSearchCV** to tune the hyperparameters.


### **Model Evaluation**
After training the model with the top features, we evaluated the performance using various metrics:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **Matthews Correlation Coefficient (MCC)**
- **AUC-ROC Curve**


### **Feature Selection Performance (Top 5, 10, 15 Features)**

Finally, we evaluated the performance of the model using the top 5, 10, and 15 features selected by the different methods:

- **Top 5 Features (K=5)**: We evaluate how well the model performs with the top 5 selected features.
- **Top 10 Features (K=10)**: We evaluate how the performance changes when we increase the number of features to 10.
- **Top 15 Features (K=15)**: We evaluate how the model performs using the top 15 features selected.

We compare the performance of the model across these different subsets of features to determine the best performing feature set.

## Results

The feature selection method using **Modified TOPSIS** showed the best performance compared to other methods. The accuracy achieved by the Random Forest model with the features selected using Modified TOPSIS is as follows:

- **k = 5**: 93.75%
- **k = 10**: 97.82%
- **k = 15**: 96.88%
  
***To check the results , run the cells in order and evaluate other metrics, as well as check compare performance with other feature selection methods. by running the notebooks***


