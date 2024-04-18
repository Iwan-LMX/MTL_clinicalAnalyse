import numpy as np
import pandas as pd
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,f1_score, roc_curve, auc,roc_auc_score,log_loss,make_scorer
from Include.MLSMOTE import get_tail_label,get_index,get_minority_instace,MLSMOTE


#---------------------------------------------------------------------------------#
# ----------------------------Self Defined Functions------------------------------#
#---------------------------------------------------------------------------------#
#Define loss function to calculate average log_loss
def loss(y_true, y_pred):
    if not isinstance(y_true, pd.DataFrame):    y_true = pd.DataFrame(y_true)
    if not isinstance(y_pred, pd.DataFrame):    y_pred = pd.DataFrame(y_pred)
    logloss = 0

    for i in range(y_true.shape[1]):  
        logloss += log_loss(y_true.iloc[:, i], y_pred.iloc[:, i], labels=[0,1])
    return logloss / y_true.shape[1]

#---------------------------------------------------------------------------------#
# --------------------------------Preprocessing DATA------------------------------#
#---------------------------------------------------------------------------------#

# Loading data
x_train = np.load('../Training data/X_train.npy') 
y_train = np.load('../Training data/y_train.npy')  
x_test=np.load("../Testing data/X_test.npy")


x_train = pd.DataFrame(x_train).rename(columns={i: f"x_{i}" for i in range(111)})
y_train = pd.DataFrame(y_train).rename(columns={i: f"y_{i}" for i in range(11)})
x_test = pd.DataFrame(x_test).rename(columns={i: f"x_{i}" for i in range(111)})

# Filling missing values with means
for column in x_train.columns:
    x_train.fillna({column: x_train[column].median()}, inplace=True)
    x_test.fillna({column: x_test[column].median()}, inplace=True)


# Filter out features that tilt largely
threshold = 0.01
cols_to_drop = [col for col in x_train.columns if (x_train[col] != 0).mean() < threshold]

x_train.drop(columns=cols_to_drop, inplace=True)
x_test.drop(columns=cols_to_drop, inplace=True)

# Standardize the input data
scaler = StandardScaler()
x_train = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns)
x_test = pd.DataFrame(scaler.fit_transform(x_test), columns=x_test.columns)

# Apply MLSMOTE
tail_labels = get_tail_label(y_train)
indices = get_index(y_train)
X_sub, y_sub = get_minority_instace(x_train, y_train)

X_resampled, y_resampled = MLSMOTE(X_sub, y_sub, 1)  # Generate new samples
X_train_final = pd.concat([x_train, X_resampled], ignore_index=True)
y_train_final = pd.concat([y_train, y_resampled], ignore_index=True)

# Split out train set and test set from original Train data
X_train, X_valid, Y_train, Y_valid = train_test_split(X_train_final, y_train_final, test_size=0.2, random_state=42)

#---------------------------------------------------------------------------------#
# -------------------------Configure SVM Model & Training-------------------------#
#---------------------------------------------------------------------------------#
loss_scorer = make_scorer(loss, greater_is_better=False)

# Create SVM model
num_labels = y_train_final.shape[1]  
svm_models = []
average_scores = {
    'f1_score': [],
    'roc_auc': []
}
predictions = pd.DataFrame()
predictions_proba = np.zeros((Y_valid.shape[0], num_labels * 2))

for i in range(num_labels):
    Y_train_label = Y_train.iloc[:, i];     Y_valid_label = Y_valid.iloc[:, i]

    # Train Model
    model = SVC(probability=True)  
    model.fit(X_train, Y_train_label)
    svm_models.append(model)

    # Make prediction
    Y_pred = model.predict(X_valid)
    Y_pred_prob = model.predict_proba(X_valid)
    predictions_proba[:, 2*i:2*i+2] = Y_pred_prob

    f1 = f1_score(Y_valid_label, Y_pred)
    roc_auc = roc_auc_score(Y_valid_label, Y_pred_prob[:, 1])

    average_scores['f1_score'].append(f1)
    average_scores['roc_auc'].append(roc_auc)

    Y_pred_test=model.predict(x_test)
    predictions[f'y_pred_{i}'] = Y_pred_test

# Calculate average evaluation
average_loss = loss(Y_valid, predictions_proba)
mean_f1 = np.mean(average_scores['f1_score'])
mean_roc_auc = np.mean(average_scores['roc_auc'])

print(f"Average F1-Score: {mean_f1}")
print(f"Average ROC-AUC: {mean_roc_auc}")
print(f"Average Loss: {average_loss}")


# Print predictions
print()
print("-----------------Predictions:-----------------")
print(predictions)





