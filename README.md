# MTL_clinicalAnalyse
This is a project about using Multitask learning method to analyze clinical trial and predicting something.


# How to develop the project

Contributors need check out a new branch to develop, you can do anything in your branch.

Via "pull requests" in Git hub to upload your code.

(You can just add the files out of your responses to gitignore. Thus they won't be sync to your branch while you push. It's up to you..)  

    :.
    │   model.ipynb     #Write MTL algorithm codes in model
    │   model.py        
    │   README.md       
    │
    ├───Separate Models 
    │   separates.ipynb #The 11 separate models to compare with MTL
    │   separates.py
    │
    ├───Testing data    #Testing data don't modify them
    │   X_test.npy
    │
    └───Training data   #Training data don't modify them
        X_train.npy
        y_train.npy
        EDA.ipynb
    
    #py file is same to ipynb, use which one you like
    #add your .vscode or any other config files to 'gitignore'

# Study approaches
## Extensive exploratory data analysis (EDA)
Since all the features in the data are anonymous, an EDA is very required.

The analysising strateges and results are shown in "EDA.ipynb".

### Design MTL Model
From study the result of EDA we decide use "xxx" algorithm as our classification learning model. 


### Design seperate sub learning models

"Wait for append"
