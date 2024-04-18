<!-- 
    Author: Iwan Li, z5433288
    Date: 2024-04-18
-->
# MTL_clinicalAnalyse
This is a project about using Multitask learning method to analyze clinical trial and predicting something.

# ⭐ How to Run it in your Machine
1. Preparing Python environment (better higher than 3.12.1)

2. Download the source code, uncompress it.

3. Maybe you should install some library including but not limited to `numpy, pandas, tensorflow, scipy, scikit-learn`.  You can just use command like `pip3 install numpy` to install one library.

4. For `EDA.ipynb`, you should choose an exist python in your machine then 'run all'.

5. For `xxx.py` files, you should use a terminal command `python xxx.py` or `python3 xxx.py` to run them.

Especially: if you need run a model please make sure that file is listed on current path. (i.e. if you need run Separate Models you need change directory to `SeparateModels`)    

# How to develop the project

Contributors need check out a new branch to develop, you can do anything in your branch.

Via "pull requests" in Git hub to upload your code.

(You can just add the files out of your responses to gitignore. Thus they won't be sync to your branch while you push. It's up to you..) 

    :.
    │   EDA.ipynb       #Extensive exploratory data analysis
    │   MTL_Model.py    #Hard Parameter Shared MTL model
    │   predicts.csv    #Prediction of MTL model
    │   README.md
    │
    ├───Include
    │   │   MLSMOTE.py
    │   │
    │   └───__pycache__
    │           MLSMOTE.cpython-312.pyc
    │
    ├───SeparateModels          #Traditional Classification Models
    │       KNN.py
    │       logistic.py
    │       naive_bayes.py
    │       RandomForest.py
    │       SVM.py
    │
    ├───Testing data
    │       X_test.npy
    │
    └───Training data
            X_train.npy
            y_train.npy
    
    #py file is same to ipynb, use which one you like
    #add your .vscode or any other config files to 'gitignore'

# 📖 Study approaches

## ⭐ Extensive exploratory data analysis (EDA)

Since all the features in the data are anonymous, an EDA is very required.

It follows traditional data analysis strategies:

1. Process data overview. (got general understanding of data we processing)

2. Check the missing and unique values.

3. Check distribution of data

4. Analyse feature skewness and kurtosis (If features of train data and test data varies a lot we should transform them)

    4.1 Transform and fix important features (The train data and test data varies a lot, it will affect final prediction)

5. Analyse types of feature values
    
The analysing techniques and results are shown in `EDA.ipynb`.

## Design MTL Model

From study the result of EDA we decide use Hard Parameter Shared Model as our Multi Task classification model.

It's shared layer very similar to Neural Network, that can well combine the input train data.

These data will be put into Task Specific layers to predict.

Here we use 4 shared layers, and 2 task specific layers to design the model.

Detail of codes are shown in `MTL_Model.py`


## Design separate sub learning models
We choose 5 popular traditional classification models to compare with MTL Model.
They are: KNN (K Nearest Neighbour), Logistic Regression, Naive Bayes, Random FOrest and SVM (Support Vector Machines)

Detail of these codes are shown in within the folder `./SeparateModels`

# Last 🎉
Thanks for [@Lingbo Ban](https://github.com/banlingbo) and [@Yishan Ma](https://github.com/Lilithys).  This project can be complete till the end can not leave with their efforts and contributions.