#head packages import
import gc
import numpy as np
import pandas as pd
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV

pd.set_option('display.max_columns' ,30) #setting as None for all row and cols
pd.set_option('display.max_rows' , 10)

train_x = np.load("../Training data/X_train.npy")
train_y = np.load("../Training data/y_train.npy")
test_x  = np.load("../Testing data/X_test.npy")

train_x = pd.DataFrame(train_x).rename(columns={i: f"x_{i}" for i in range(111)})
train_y = pd.DataFrame(train_y).rename(columns={i: f"y_{i}" for i in range(11)})

#1. Data overview
test_x = pd.DataFrame(test_x).rename(columns={i: f"x_{i}" for i in range(111)})
print(pd.concat([train_x.head(), train_x.tail()]))
print(train_y)
print(train_x.describe())
print(train_x.info())

#2. Check missing and unique values
cols = [f"x_{i}" for i in range(111)]
data = pd.concat([train_x, test_x])

tmp = pd.DataFrame()
tmp['count'] = data[cols].count().values
tmp['missing_rate'] = (data.shape[0] - tmp['count']) / data.shape[0]
tmp['nunique'] = data[cols].nunique().values
tmp.index = cols
print(tmp)

sns.violinplot(np.log(train_x["x_1"]))

#3. Check distribution of training data
sns.displot(train_x["x_1"])
sns.displot(np.log(train_x["x_1"]))


#4. Analyse feature skewness and kurtosis
# Training set and testing set vary from skewness and kurtosis at some places
pd.set_option('display.max_rows' , 10)
tmp = pd.DataFrame(index = cols)
for col in cols:
    tmp.loc[col, 'train_Skewness'] = train_x[col].skew()
    tmp.loc[col, 'test_Skewness'] = test_x[col].skew()
    tmp.loc[col, 'train_Kurtosis'] = train_x[col].kurt()
    tmp.loc[col, 'test_Kurtosis'] = test_x[col].kurt()
print(tmp)


# Analyse correlation of features and every y_j
data_train = pd.concat([train_x, train_y], axis=1)
correlations = pd.DataFrame()
for c in train_y.columns:
    correlations[c] = data_train[list(train_x.columns) + [c]].corr()[c]
    correlations[c].sort_values(ascending=False).head()


# 4.1 Transform to fix important features
# Filled up missing values
for column in train_x.columns:
    train_x.fillna({column: train_x[column].median()}, inplace=True)
    test_x.fillna({column: test_x[column].median()}, inplace=True)

# filter out less valued columns
threshold = 0.01  
cols_to_drop = [col for col in train_x.columns if (train_x[col] != 0).mean() < threshold]
cols_to_drop += ["x_8",  "x_26", "x_29", "x_30", "x_32", "x_39", "x_50", "x_53", "x_54", "x_64", "x_68", "x_71", "x_93", "x_102"]

train_x.drop(columns=cols_to_drop, inplace=True)
test_x.drop(columns=cols_to_drop, inplace=True)

from scipy.stats import yeojohnson
# Yeo-Johnson transform
train_log = train_x.apply(lambda x: yeojohnson(x)[0])
test_log = test_x.apply(lambda x: yeojohnson(x)[0])


train_skewness = train_log.apply(lambda x: x.skew())
test_skewness = test_log.apply(lambda x: x.skew())
train_kurtosis = train_log.apply(lambda x: x.kurt())
test_kurtosis = test_log.apply(lambda x: x.kurt())

summary = pd.DataFrame({
    'train_skewness': train_skewness,
    'test_skewness': test_skewness,
    'train_kurtosis': train_kurtosis,
    'test_kurtosis': test_kurtosis
})

# Define threshold
skew_threshold = 1.0
kurtosis_threshold = 5.0

pd.set_option('display.max_rows' , None)


large_diff_skew = summary[(abs(summary['train_skewness'] - summary['test_skewness']) > skew_threshold)]
large_diff_kurtosis = summary[(abs(summary['train_kurtosis'] - summary['test_kurtosis']) > kurtosis_threshold)]

# Check differences
print("Columns with large skewness differences:")
print(large_diff_skew)

print("\nColumns with large kurtosis differences:")
print(large_diff_kurtosis)


# Analyse correlation of features and every y_j
data_train = pd.concat([train_x, train_y], axis=1)
correlations = pd.DataFrame()
for c in train_y.columns:
    correlations[c] = data_train[list(train_x.columns) + [c]].corr()[c]
    correlations[c].sort_values(ascending=False).head()




# 5. Analyse types of feature values
# The print out is features amount of values in type 
multitags = []
for col in list(train_x.columns):
    col, len(train_x[col].value_counts())
    if len(train_x[col].value_counts()) > 20:
        multitags.append(col)