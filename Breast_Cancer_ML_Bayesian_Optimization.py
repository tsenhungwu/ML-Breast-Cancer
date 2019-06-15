# Packages
# Data Structures.
import numpy as np
import pandas as pd
import csv
import json
from time import time
# Dataset preprocessing.
from Breast_Cancer_Functions import final_preprocess
# Machine Learning Algorithms
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb 
# Bayesian Optimization: Hyperparameter tuning
from hyperopt import hp
from hyperopt import Trials
from hyperopt import tpe
from timeit import default_timer as timer
from hyperopt import STATUS_OK
from hyperopt import fmin
# Model performance evaluation.
from sklearn.model_selection import cross_validate
# Command lines.
import argparse

# Function I: Customize command lines.
def command_line():
    '''
    Input:
        No input needed.
        dataset_dir: A dataset directory where it stores a dataset which is going to be learned by a Machine Learning algorithm.
        --model: A Machine Learning algorithm name ('GradientBoostingClassifier' or 'XGBClassifier').
        --BO_iter: Number of iterations for the 1st Bayesian optimization. Default number is 10.
        --BO_save_dir: A directory where the program saves optimization history as a csv file.
        --BO_json_save_dir: A directory where the program saves optimization history as a json file (For later continued Bayesian Optimization).
        --continue_BO: Whether perform continued Bayesian Optimization based on a previous optimization history.
        --BO_cont_iter: Number of iterations for the continued Bayesian optimization. Default number is 10.
    Output:
        Return all arguments.
    '''
    parser = argparse.ArgumentParser(description='Breast_Cancer')
    # A dataset directory.
    parser.add_argument('dataset_dir', type=str, help='dataset directory')
    # Specify which model to be optimized.
    parser.add_argument('--model', type=str, dest='model_name', default='GradientBoostingClassifier', help='Choose a ML algorithm')
    # Number of iterations in Bayesian Optimization.
    parser.add_argument('--BO_iter', type=int, dest='optimization_iter', default=10, help='Number of iterations in Bayesian Optimization')
    # A csv file directory stores Bayesian Optimization searching history.
    parser.add_argument('--BO_save_dir', type=str, dest='BO_saved_dir', default='GB_clf.csv', help='Bayesian Optimization searching history csv directory')
    # A json file directory stores Bayesian Optimization Trials object (searching history).
    parser.add_argument('--BO_json_save_dir', type=str, dest='BO_json_saved_dir', default='GB_clf.json', help='Bayesian Optimization searching history json directory')
    # Continue executing Bayesian Optimization based on previous optimization searching history.
    parser.add_argument('--continue_BO', type=bool, dest='continue_BO', default=False, help='True: execute Bayesian Optimization again')
    # Number of iterations in continued Bayesian Optimization.
    parser.add_argument('--BO_cont_iter', type=int, dest='optimization_cont_iter', default=10, help='Number of iterations in continued Bayesian Optimization')
    
    # Output
    args = parser.parse_args()
    return args

# Function II
def searching_space(model_name):
    # Specify searching space of hyperparameters.
    if model_name == 'GradientBoostingClassifier':   
        space = {'n_estimators': hp.randint('n_estimators', 500),
                 'max_depth': hp.uniform('max_depth', 0, 8),
                 'subsample': hp.uniform('subsample', 0.0, 1.0),
                 'max_features': hp.uniform('max_features', 0.0, 1.0),
                 'learning_rate': hp.uniform('learning_rate', 0.0, 1.0),
                 'random_state': 0
                }
    elif model_name == 'XGBClassifier':
        space = {'objective': 'binary:logistic',
                 'eta': hp.uniform('eta', 0.0, 1.0),
                 'max_depth': hp.randint('max_depth', 8),
                 'subsample': hp.uniform('subsample', 0.0, 1.0),
                 'colsample_bytree': hp.uniform('colsample_bytree', 0.0, 1.0),
                 'colsample_bylevel': hp.uniform('colsample_bylevel', 0.0, 1.0),
                 'colsample_bynode': hp.uniform('colsample_bynode', 0.0, 1.0),
                 'lambda': hp.uniform('lambda', 0.0, 1.0),
                 'alpha': hp.uniform('alpha', 0.0, 1.0),
                 'num_round': hp.randint('num_round', 500),
                 'task': 'train'
                }
    else:
        print("Can't initiate a model successfully, please try other names. (E.g. GradientBoostingClassifier)")
        
    return space

# Function III
# Element II: Define the objective for the optimization.
# Here is to minimize the negative f1 score since it's 
# a metric that balances between precision and recall.
def gb_clf_objective(hyperparameters):
    
    # Keep track of evals
    global ITERATION
    
    ITERATION += 1
    
    start = timer()
    
    # Perform 5-fold cross validation
    model = GradientBoostingClassifier(**hyperparameters)
    global train_standardized_data
    cv_results = cross_validate(model, train_standardized_data, y_train, 
                                scoring='f1', cv=5, 
                                return_train_score=False)
    run_time = timer() - start
    
    # Loss must be minimized (put a negative sign)
    loss = -(np.mean(cv_results['test_score']))

    # Write searching results to a csv file ('a' means append)
    of_connection = open(OUT_FILE, 'a')
    writer = csv.writer(of_connection)
    writer.writerow([loss, hyperparameters, ITERATION, run_time])
    of_connection.close()

    # Dictionary with information for later evaluations
    return {'loss': loss, 'hyperparameters': hyperparameters, 'iteration': ITERATION,
            'train_time': run_time, 'status': STATUS_OK}

def xgb_clf_objective(hyperparameters):
    
    # Keep track of evals
    global ITERATION
    
    ITERATION += 1
    
    start = timer()
    
    # Perform 5-fold cross validation
    model = xgb.XGBClassifier(**hyperparameters)
    global train_standardized_data
    cv_results = cross_validate(model, train_standardized_data, y_train, scoring='f1', cv=5, return_train_score=False)
    run_time = timer() - start
    
    # Loss must be minimized (put a negative sign)
    loss = -(np.mean(cv_results['test_score']))

    # Write searching results to a csv file ('a' means append)
    of_connection = open(OUT_FILE, 'a')
    writer = csv.writer(of_connection)
    writer.writerow([loss, hyperparameters, ITERATION, run_time])
    of_connection.close()

    # Dictionary with information for later evaluations
    return {'loss': loss, 'hyperparameters': hyperparameters, 'iteration': ITERATION,
            'train_time': run_time, 'status': STATUS_OK}


# Function IV       
def bayesian_optimize(model_name, n_eval):
    
    # Record searching results.
    trials = Trials()
    
    # Create a csv file to store results.
    of_connection = open(OUT_FILE, 'w')
    writer = csv.writer(of_connection)

    global ITERATION
    ITERATION = 0
    
    # Write column names in the file.
    headers = ['loss', 'hyperparameters', 'iteration', 'runtime']
    writer.writerow(headers)
    of_connection.close()

    # Start optimization!
    if model_name == 'GradientBoostingClassifier':
        best = fmin(fn = gb_clf_objective,
                    space = searching_space(model_name=model_name), 
                    algo = tpe.suggest, 
                    trials = trials,
                    max_evals = n_eval)
    elif model_name == 'XGBClassifier':
        best = fmin(fn = xgb_clf_objective,
                    space = searching_space(model_name=model_name), 
                    algo = tpe.suggest, 
                    trials = trials,
                    max_evals = n_eval)
    
    return best, trials

 # Function V
def continue_bayesian_optimization(trail, model_name, n_eval):
    # Start optimization!
    trials = Trials(trail)

    if model_name == 'GradientBoostingClassifier':
        best = fmin(fn = gb_clf_objective,
                    space = searching_space(model_name=model_name), 
                    algo = tpe.suggest, 
                    trials = trials,
                    max_evals = n_eval)
    elif model_name == 'XGBClassifier':
        best = fmin(fn = xgb_clf_objective,
                    space = searching_space(model_name=model_name), 
                    algo = tpe.suggest, 
                    trials = trials,
                    max_evals = n_eval)
    return best, trials

# Main executions.
def main():
    # Command lines.
    # dataset_path: Dataset directory
    # baye_optim_save_path: A csv file stores Bayesian Optimization searching history.
    # baye_optim_save_json_path: A json file stores Bayesian Optimization searching history.
    # model_name: A Machine Learning algorithm name (E.g. 'GradientBoostingClassifier' or 'GradientBoostingClassifier').
    # optimization_iter: Number of iterations of Bayesian Optimization.
    # continue_baye_optimization: True or False (False to perform 1st Bayesian Optimization).
    # cont_optimization_iter: Number of iterations of continued Bayesian Optimization.

    args = command_line()
    dataset_path = args.dataset_dir
    baye_optim_save_path = args.BO_saved_dir
    baye_optim_save_json_path = args.BO_json_saved_dir
    model_name = args.model_name
    optimization_iter = args.optimization_iter
    continue_baye_optimization = args.continue_BO
    cont_optimization_iter = args.optimization_cont_iter

    # Read in and process the dataset.
    data = pd.read_csv(dataset_path)
    global train_standardized_data, y_train
    train_standardized_data, test_standardized_data, y_train, y_test = final_preprocess(data=data)
    print('Processed dataset...success!')

    global OUT_FILE
    OUT_FILE = baye_optim_save_path

    # Continue Bayesian Optimization.
    if continue_baye_optimization:
        with open(baye_optim_save_json_path) as json_file:
            cont_trail = json.load(json_file)
        global ITERATION
        ITERATION = len(cont_trail)
        best, new_trials = continue_bayesian_optimization(model_name=model_name, n_eval=cont_optimization_iter, trail=cont_trail)
        new_trials_dict = sorted(new_trials.results, key = lambda x: x['loss'])
        for each in new_trials_dict:
            cont_trail.append(each)
        with open(baye_optim_save_json_path, 'w') as f:
            json.dump(cont_trail, f)
    # Start 1st Bayesian Optimization.
    # After finished Bayesian Optimization, the program automatically saves the optimization searching history into 'OUT_FILE', a csv file.
    else:
        best, trials = bayesian_optimize(model_name=model_name, n_eval=optimization_iter)
        # Also, save the optimization searching history into a .json file for the purpose of continued Bayesian Optimization.
        trials_dict = sorted(trials.results, key = lambda x: x['loss'])
        with open(baye_optim_save_json_path, 'w') as f:
            f.write(json.dumps(trials_dict))


# Notice!
# In terminal, type 'python Breast_Cancer_ML_Bayesian_Optimization.py data.csv' provides default functionalities:
#                   (1) Performs 10 Bayesian Optimization iterations on GradientBoostingClassifier and 
#                   (2) Saves optimization searching result in GB_clf.csv and GB_clf.json (.json for continued Bayesian Optimization purpose) 
#                       under the same directory as this program.
# Notice: If for the 1st Bayesian Optimization, the program can't execute functionalities for continued Bayesian Optimization since it needs previous searching history.
# Other optional functionalities by typing below command lines after 'python Breast_Cancer_Functions.py data.csv':
#                   (1) --BO_save_dir "put the directory where you want to save optimization result as a csv file" 
#                   (2) --model "put another ML algorithm name" 
#                   (3) --BO_iter "number of iterations in Bayesian Optimization" (the larger number is, the longer it takes)
#                   (4) --BO_json_save_dir "put the directory where you want to save optimization result as a json file" 
#                   (5) --continue_BO "If want to perform more optimizations, put True"
#                   (6) --BO_cont_iter "Number of continued Bayesian Optimization"

if __name__ == '__main__':
    main()


