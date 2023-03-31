import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, DotProduct
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from matplotlib import pyplot as plt
import pickle
import argparse


parser = argparse.ArgumentParser(description="Implementation of River Sensor Model")

#########################
#### data parameters ####
#########################
parser.add_argument("--data_path", type=str, default=".",
                    help="path to dataset repository")

parser.add_argument("--saving_dir", type=str, default=".",
                    help="experiment dump path for saving models and log")

parser.add_argument("--random_state", type=str, default=1,
                    help="random state for experiment" )

def test_pickle(model_dict,pickled_models_dict,eval_dict,pickled_eval_dict, df,random_state=1):
    assert eval_dict == pickled_eval_dict
    targets = df.columns.to_list()
    for target in targets:
        features = df.drop(columns = [target])
        labels = df[target]

        normalized_features=(features-features.mean())/features.std()

        X = normalized_features.to_numpy()
        y = labels.to_numpy()
        X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=random_state)
        for architecture in model_dict[target].keys():
            assert all(model_dict[target][architecture].predict(X_test) == pickled_models_dict[target][architecture].predict(X_test))
            assert all(model_dict[target][architecture].predict(X_train) == pickled_models_dict[target][architecture].predict(X_train))
    print('Pickled Tests Passed')

def hyper_optimise_mlp(X_tr,y_tr,random_state = 1):
    estimator = MLPRegressor(random_state=random_state, max_iter=5000)
    
    param_grid = {'hidden_layer_sizes': [(4,),(7),(13,),(25,),(50,),(100,)],
            'activation': ['relu','tanh'],
            }

    gsc = GridSearchCV(
        estimator,
        param_grid,
        cv=5, scoring='neg_root_mean_squared_error', verbose=3, n_jobs=-1)

    grid_result = gsc.fit(X_tr, y_tr)

    return gsc

def hyper_optimise_rf(X_tr,y_tr,random_state = 1):
    estimator = RandomForestRegressor(random_state=random_state)

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    param_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}

    gsc = GridSearchCV(
        estimator,
        param_grid,
        cv=5, scoring='neg_root_mean_squared_error', verbose=3, n_jobs=-1)

    grid_result = gsc.fit(X_tr, y_tr)

    return gsc

def hyper_optimise_gpr(X_tr,y_tr,random_state = 1):
    estimator = GaussianProcessRegressor(random_state=random_state, max_iter=5000)
    
    param_grid = [{
        "alpha":  [1e-2, 1e-3],
        "kernel": [RBF(l) for l in np.logspace(-1, 1, 2)]
    }, {
        "alpha":  [1e-2, 1e-3],
        "kernel": [DotProduct(sigma_0) for sigma_0 in np.logspace(-1, 1, 2)]
    }]

    gsc = GridSearchCV(
        estimator,
        param_grid,
        cv=5, scoring='neg_root_mean_squared_error', verbose=3, n_jobs=-1)

    grid_result = gsc.fit(X_tr, y_tr)

    return gsc

def hyper_optimise_lasso(X_tr,y_tr,random_state = 1):
    estimator = Lasso(random_state=random_state, max_iter=5000)
    
    param_grid = {
        "alpha":  np.arange(0, 1, 0.01),
    }

    gsc = GridSearchCV(
        estimator,
        param_grid,
        cv=5, scoring='neg_root_mean_squared_error', verbose=3, n_jobs=-1)

    grid_result = gsc.fit(X_tr, y_tr)

    return gsc

def pred25(y_true, y_pred):
    perc = np.abs(y_true-y_pred)/y_true
    return np.count_nonzero(perc < 0.25)/len(y_true)

def evaluate(estimator, X_test,y_test):
    y_predicted = estimator.predict(X_test)
    rmse = mean_squared_error(y_test, y_predicted, squared=False)
    coef_det = estimator.score(X_test, y_test)
    rmsle = np.log(rmse)
    pred_25 = pred25(y_test, y_predicted)
    mmre = mean_absolute_percentage_error(y_test, y_predicted)
    evaluation = {}
    evaluation['coef_det'] = coef_det
    evaluation['rmse'] = rmse
    evaluation['rmsle'] = rmsle
    evaluation['mmre'] = mmre
    evaluation['pred25'] = pred_25
    evaluation['errors'] = y_test - y_predicted
    return evaluation


def train_all_models(df,random_state=1):
    models = {}
    evaluations = {}
    targets = df.columns.to_list()
    for target in targets:
        print("Training {} model ...".format(target))
        features = df.drop(columns = [target])
        labels = df[target]

        normalized_features=(features-features.mean())/features.std()

        X = normalized_features.to_numpy()
        y = labels.to_numpy()
        X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=random_state)
        
        architectures = {}
        architecture_evaluations = {}
        
        mlp_regressor = hyper_optimise_mlp(X_train,y_train,random_state=random_state)
        architectures['MLPRegressor'] = mlp_regressor
        architecture_evaluations['MLPRegressor'] = evaluate(mlp_regressor, X_test,y_test)
        
        rf_regressor = hyper_optimise_rf(X_train,y_train,random_state=random_state)
        architectures['RFRegressor'] = rf_regressor
        architecture_evaluations['RFRegressor'] = evaluate(rf_regressor, X_test,y_test)
        
        gp_regressor = hyper_optimise_gpr(X_train,y_train,random_state=random_state)
        architectures['GPRegressor'] = gp_regressor
        architecture_evaluations['GPRegressor'] = evaluate(gp_regressor, X_test,y_test)
        
        lasso_regressor = hyper_optimise_lasso(X_train,y_train,random_state=random_state)
        architectures['LassoRegressor'] = lasso_regressor
        architecture_evaluations['LassoRegressor'] = evaluate(lasso_regressor, X_test,y_test)
        
        
        models[target] = architectures
        evaluations[target] = architecture_evaluations
    return models,evaluations

def main():
    print("Starting Main")
    args = parser.parse_args()
    DATA_PATH = args.data_path
    SAVING_DIR = args.saving_dir
    random_state = int(args.random_state)
    print("Using random state:",random_state)

    df = pd.read_csv(DATA_PATH)

    
    print("Loaded Data")
    print("Starting Experiments")
    models, evaluations = train_all_models(df,random_state=random_state)

    models_path = SAVING_DIR +'/models.pickle'
    evaluations_path = SAVING_DIR + '/evaluations.pickle'

    with open(models_path, 'wb') as handle:
        pickle.dump(models, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(evaluations_path, 'wb') as handle:
        pickle.dump(evaluations, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(models_path, 'rb') as handle:
        models_load = pickle.load(handle)

    with open(evaluations_path, 'rb') as handle:
        evaluations_load = pickle.load(handle)

    test_pickle(models,models_load,evaluations,evaluations_load, df,random_state=random_state)

if __name__ == "__main__":
    main()