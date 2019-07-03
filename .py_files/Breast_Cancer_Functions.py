# Packages
# Data Structures.
import numpy as np
import pandas as pd
from time import time
import json
# Visualization
import matplotlib.pyplot as plt
# Statistical Hypothesis testing.
from scipy.stats import f
# Dataset preprocessing.
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# Machine Learning algorithms.
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
# Model performance evaluation.
from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score, accuracy_score, roc_auc_score, roc_curve

# Function I: Drop off irrelevant columns in the dataset.
def preprocess_dataset(data):
    '''
    Drop off irrelevant columns in the dataset.

    Input: 
        data: a pandas dataframe object.
    
    Output: 
        data: return a pandas dataframe object without 'id' and 'Unnamed: 32' columns. 
    '''
    data.drop(columns=['id','Unnamed: 32'], inplace = True)
    return data

# Function II: Plot pair-plot to inspect relationship between predictors and the response variable.
def relationship_pairplot(data, feature, outcome):
    '''
    Plot pair-plot to inspect relationship between predictors and the response variable.

    Inputs:
        data: a pandas dataframe object.
        feature: a list object contains predictors' names.
        outcome: a string type variable, which is the response variable name.
        
    Output:
        The function doesn't return anything. It directly plots out N by N scatter plots and line plots 
        located at diagonal lines.
    '''
    space = feature + [outcome]
    sb.pairplot(data=data[space], palette='Set1', hue=outcome)

# Function III: Perform an F-test to compare two sample variances.
def f_test_two_p_variance(var1, var2, n1, n2, alpha_level):
    '''
    Perform an F-test for comparing variances between two samples.
    
    Inputs:
        var1: the variance of the population 1.
        var2: the variance of the population 2.
        n1: sample size of the population 1.
        n2: sample size of the population 2.
        alpha_level: an alpha threshold for rejecting Null hypothesis. (can be 0.1, 0.05, or 0.1)

    Outputs: 
        F_ratio: the ratio of variances between two populations.
        p-value: p-value.
        (lower, upper): confidence intervals given an alpha level.
    '''
    # F-test statistic
    F_ratio = var1 / var2
    
    # p-value: (two-sided test: the interest is to compare whether two variances are equal)
    p_value = min(f.cdf(F_ratio, n1-1, n2-1), 1 - f.cdf(F_ratio, n1-1, n2-1))
    p_value = p_value*2

    # 95% C.I. if alpha_level = 0.05.
    upper = 1 / (f.ppf(alpha_level/2, n1-1, n2-1)) * F_ratio
    lower = 1 / (f.ppf(1 - (alpha_level/2), n1-1, n2-1)) * F_ratio
    
    # Outputs
    return round(F_ratio, 4), round(p_value, 4) , (round(lower, 4), round(upper,4 ))

# Function IV: Perform all required processing tasks including dropping irrelevant columns, label encoding on 'diagnosis' column,
# splitting data into training and testing sets, and standardization on the dataset.
def final_preprocess(data):
    '''
    Perform all required processing tasks including dropping irrelevant columns, label encoding on 'diagnosis' column,
    splitting data into training and testing sets, and standardization on the dataset.

    Input:
        data: a pandas dataframe object.
    
    Outputs:
        train_standardized_data: a standardized training feature set.
        test_standardized_data:  a standardized testing feature set.
        y_train: a training response set.
        y_test: a testing response set.
        x_train.columns: provide column names for later feature importance reference.
    '''
    # Drop off columns.
    data.drop(columns=['id','Unnamed: 32'], inplace = True)

    # Label transformation on response variable, 'diagnosis'.
    le = LabelEncoder()
    le.fit(data['diagnosis'])
    data['diagnosis'] = le.transform(data['diagnosis'])
    
    # Split data into training & testing sets (80% & 20% splitting).
    x_train, x_test, y_train, y_test = train_test_split(
        data[data.columns.difference(['diagnosis'])], data.diagnosis,  test_size=0.2, random_state=0) 

    # Standardization.
    # Notice: Fit a standardization scaler 'only' on training set and 'transform' on training and testing sets.
    std_scaler = StandardScaler().fit(x_train)
    train_standardized_data = std_scaler.transform(x_train)
    test_standardized_data = std_scaler.transform(x_test)

    # Outputs.
    return train_standardized_data, test_standardized_data, y_train, y_test, x_train.columns

# Function V: Fit a model on the training set and test it on the testing set.
def train_predict(learner, X_train, y_train, X_test, y_test): 
    '''
    Fit a model on the training set and test it on the testing set.

    inputs:
        learner: a machine learning algorithm to be trained and predicted on.
        X_train: features training set.
        y_train: response training set.
        X_test: features testing set.
        y_test: response testing set.
    
    Output:
        results: a dictionary object contains six keys (train_time, pred_time acc_train, acc_test, f_train, f_test) and 
        six corresponding values.
    '''
    # Collect results.
    results = {}
    
    # Fit training data on a classifier.
    start = time()
    learner = learner.fit(X_train, y_train)
    end = time() 
    
    # Record training time.
    results['train_time'] = end-start
        
    # Make predictions on training and testing data.
    start = time()
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train)
    end = time()
    
    # Record the total prediction time.
    results['pred_time'] = end - start
            
    # Compute accuracy on training and testing data.
    results['acc_train'] = accuracy_score(y_train, predictions_train)
    results['acc_test'] = accuracy_score(y_test, predictions_test)
    
    # Compute F-score on training and testing data.
    results['f_train'] = fbeta_score(y_train, predictions_train, beta = 1.0)
    results['f_test'] = fbeta_score(y_test, predictions_test, beta = 1.0)
       
    # Success!
    print("{} trained on {} samples.".format(learner.__class__.__name__, X_train.shape[0]))
        
    # Output.
    return results

# Function VI: find the optimal combination of the hyperparameters.
def best_hyperparameter(json_data):
    '''
    Search the optimal model performance a machine learning model achieved and the assoicated hyperparameters
    from a json file containing a hyperparameter searching history.

    Input:
        json_data: a json file directory where it stores hyperparameter searching history.

    Output:
        best_model: a dictionary object contains two keys, 'loss' stores the optimal model performance and 
        'hyperparameter' stores the corresponding hyperparameters to achieve the model performance.
    '''
    with open(json_data) as json_file:
        searching_history = json.load(json_file)
    
    best_performance = 0
    best_model = {}

    for each in searching_history:
        if each['loss'] < best_performance:
            best_model['loss'] = each['loss']
            best_model['hyperparameter'] = each['hyperparameters']
            best_performance = best_model['loss']
    return best_model

# Function VII: print metric scores on training and testing sets achieved by a machine learning model.   
def get_performance_metrics(y_train, y_train_pred, y_test, y_test_pred, threshold=0.5):
    '''
    Print scores including auc, accuracy, precision, recall, f1 score of a machine learning model. 
    Inputs:
        y_train: response variables in a training set.
        y_train_pred: response variables predicted by a machine learning model.
        y_test: response variables in a testing set.
        y_test_pred: response variables predicted by a machine learning model.
        threshold: a probability threshold value for determining response variables (default value is 0.5).

    Output:
        No output returned. 
        The function prints out performance report as a pandas dataframe object.
    '''
    metric_names = ['AUC','Accuracy','Precision','Recall','f1-score']
    metric_values_train = [roc_auc_score(y_train, y_train_pred),
                           accuracy_score(y_train, y_train_pred>threshold),
                           precision_score(y_train, y_train_pred>threshold),
                           recall_score(y_train, y_train_pred>threshold),
                           f1_score(y_train, y_train_pred>threshold)
                          ]
    metric_values_test = [roc_auc_score(y_test, y_test_pred),
                          accuracy_score(y_test, y_test_pred>threshold),
                          precision_score(y_test, y_test_pred>threshold),
                          recall_score(y_test, y_test_pred>threshold),
                          f1_score(y_test, y_test_pred>threshold)
                         ]
    all_metrics = pd.DataFrame({'metrics':metric_names,
                                'train':metric_values_train,
                                'test':metric_values_test},
                               columns=['metrics','train','test']).set_index('metrics')
    print(all_metrics)

# Function VIII: plot a roc curve.
def plot_roc_curve(y_train, y_train_pred, y_test, y_test_pred):
    '''
    Plot two roc curves seperately on training and testing set.

    Inputs:
        y_train: response variables in a training set.
        y_train_pred: response variables predicted by a machine learning model.
        y_test: response variables in a testing set.
        y_test_pred: response variables predicted by a machine learning model.

    Output:
        No output returned.
        The function plots out two roc curves based on the inputs it received.
    '''

    roc_auc_train = roc_auc_score(y_train, y_train_pred)
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_pred)

    roc_auc_test = roc_auc_score(y_test, y_test_pred)
    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_pred)
    plt.figure()
    lw = 2
    plt.plot(fpr_train, tpr_train, color='green',
             lw=lw, label='ROC Train (AUC = %0.4f)' % roc_auc_train)
    plt.plot(fpr_test, tpr_test, color='darkorange',
             lw=lw, label='ROC Test (AUC = %0.4f)' % roc_auc_test)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig('test.png')
    plt.show()

# Function VIIII: reduce feature space.
def feature_dimension_reduction(data):
    '''
    Filter top features and complete the same processing tasks as final_preprocess function.

    Input:
        data: a pandas dataframe object.
    
    Outputs:
        train_standardized_data: a standardized training feature set.
        test_standardized_data:  a standardized testing feature set.
        y_train: a training response set.
        y_test: a testing response set.
        x_train.columns: provide column names for later feature importance reference.
    '''

    # Select top features which contributed to 90% total feature importance in the dataset.
    columns = ['concave points_worst', 'area_se', 'area_mean', 'area_worst', 'diagnosis']
    data = data[columns]

    # Label transformation on response variable, 'diagnosis'.
    le = LabelEncoder()
    le.fit(data['diagnosis'])
    data['diagnosis'] = le.transform(data['diagnosis'])

    # Split data into training & testing sets (80% & 20% splitting).
    x_train, x_test, y_train, y_test = train_test_split(
        data[data.columns.difference(['diagnosis'])], data.diagnosis,  test_size=0.2, random_state=0) 

    # Standardization.
    # Notice: Fit a standardization scaler 'only' on training set and 'transform' on training and testing sets.
    std_scaler = StandardScaler().fit(x_train)
    train_standardized_data = std_scaler.transform(x_train)
    test_standardized_data = std_scaler.transform(x_test)

    # Outputs.
    return train_standardized_data, test_standardized_data, y_train, y_test, x_train.columns