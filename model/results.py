import pandas as pd
import numpy as np
import sklearn
# from sklearn.neural_network import MLPRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import cross_val_score
from matplotlib import pyplot as plt
import pickle
import os

import argparse

parser = argparse.ArgumentParser(description="Results of River Sensor Model Experiments")

#########################
#### data parameters ####
#########################
parser.add_argument("--data_path", type=str, default="/home-mscluster/erex/sensors_project/data/river_sensor_data.csv",
                    help="path to dataset repository")

parser.add_argument("--experiments_path", type=str, default="/home-mscluster/erex/sensors_project/model/experiments",
                    help="path to experiment saving directories")

parser.add_argument("--results_path", type=str, default="/home-mscluster/erex/sensors_project/model/results",
                    help="path to results saving directory")

def is_saving_dir(directory):
    directory = str(directory)
    # directory = directory.split('/')[-1]
    if "saving_dir" in directory:
        return True
    else: return False

def get_experiment_id(saving_dir):
    path = str(saving_dir)
    directory = path.split('/')[-1]
    last_char = directory[-1]
    experiment_id = int(last_char)
    return experiment_id

def get_experiments(EXPERIMENTS_PATH):
    rootdir = EXPERIMENTS_PATH
    experiments = []

    for subdir, dirs, files in os.walk(rootdir):
        if is_saving_dir(subdir):
            experiment = {}
            experiment["id"] = get_experiment_id(subdir)
            for file in files:
                path  = os.path.join(subdir, file)
                with open(path, 'rb') as pickle_file:
                    if file.endswith("evaluations.pickle"):
                        experiment['evaluations']  = pickle.load(pickle_file)
                    elif file.endswith("models.pickle"):
                        experiment['models']  = pickle.load(pickle_file)
                    # else:
                    #     print('unacceptable file',file)
            # for key in experiment['evaluations'].keys():
            #     experiment['evaluations'][key]['rmse'] = np.exp(experiment['evaluations'][key]['rmsle'])  # There was an error in main.py, this resolves it
            
            experiments.append(experiment)
            architecture = list(experiment['models'][list(experiment['models'].keys())[0]].keys())[0]
        # else:
        #     print('not a savingdir',subdir)
    return experiments, architecture

def get_median_target_experiment(target, experiments, architecture):
    target_experiments = [] 
    for i in range(len(experiments)):
        target_experiment = {
            "id":experiments[i]["id"],
            "evaluations":experiments[i]["evaluations"][target][architecture],
            "models":experiments[i]["models"][target][architecture]
            }

        target_experiments.append(target_experiment)

    sorted_target_experiments = sorted(target_experiments, key=lambda d: d['evaluations']['rmse']) 
    median_target_experiment = sorted_target_experiments[int(len(sorted_target_experiments)//2)]
    return median_target_experiment

def get_mean_rmse(target,experiments,architecture):
    rmse_sum = 0.0
    for i in range(len(experiments)):
        rmse_sum += experiments[i]["evaluations"][target][architecture]['rmse']

    rmse_avg = rmse_sum/len(experiments)
    return rmse_avg

def filenameify(s):
    return "".join(x for x in s if x.isalnum())

def histogram(RESULTS_SAVING_PATH, target, errors):
    
    fig = plt.figure(figsize =(10, 7))
    
    counts, bins = np.histogram(np.abs(errors),bins=100)
    plt.stairs(counts, bins)

    plt.title("{} Error Histogram".format(target))
        
    save_path = RESULTS_SAVING_PATH+'/'+filenameify(target)+'.png'
    
    plt.savefig(save_path, bbox_inches='tight')

def main():
    print("Starting Main")
    args = parser.parse_args()
    EXPERIMENTS_PATH = args.experiments_path
    DATA_PATH = args.data_path
    RESULTS_SAVING_PATH = args.results_path

    df = pd.read_csv(DATA_PATH)

    targets = df.columns.to_list()

    experiments, architecture = get_experiments(EXPERIMENTS_PATH)
    
    best_rmse = None
    best_target=None
    
    save_path = RESULTS_SAVING_PATH+'/'+architecture+'.txt'
    
    with open(save_path, "w") as f:
        print("Experiments Results for {} architecture".format(architecture), file=f)
        print("####################################################################################################", file=f)
    
    for target in targets:
        median_target_experiment = get_median_target_experiment(target, experiments, architecture)
        
        median_evaluation_metrics = {key: median_target_experiment["evaluations"][key] for key in median_target_experiment["evaluations"] if key != 'errors'}
        mean_rmse = get_mean_rmse(target,experiments,architecture)
        # find out which target we perform best for
        if best_rmse is None:
            best_rmse = mean_rmse
            best_target = target
        elif best_rmse >= mean_rmse:
            best_rmse = mean_rmse
            best_target = target
            
        
        with open(save_path, "a") as f:
            print("Target:",target, file=f)
            print("Optimised Hyper-Parameters for median model:",median_target_experiment["models"].best_params_, file=f)
            print("Median Evaluation Metrics:",median_evaluation_metrics, file=f)
            print("Mean RMSE",mean_rmse, file=f)
            print("####################################################################################################", file=f)
        
        errors = median_target_experiment["evaluations"]['errors']
        histogram(RESULTS_SAVING_PATH, target, errors)
        
    with open(RESULTS_SAVING_PATH+'/best_target.txt', "a") as f:
        print("Best target:", best_target, file=f)
        print("rmse:", best_rmse, file=f)
    



if __name__ == "__main__":
    main()