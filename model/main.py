import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
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

        assert all(model_dict[target].predict(X_test) == pickled_models_dict[target].predict(X_test))
        assert all(model_dict[target].predict(X_train) == pickled_models_dict[target].predict(X_train))
    print('Pickled Tests Passed')

def hyper_optimise(X_tr,y_tr,random_state = 1):
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

        models[target] = hyper_optimise(X_train,y_train,random_state=random_state)

        evaluations[target] = evaluate(models[target], X_test,y_test)
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