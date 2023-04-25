# River Sensors ML Research
Research into the efficacy of using ML regression techniques to predict the output of water quality sensors.

We use slurm to run our experiments. See the [sbatch files](https://github.com/evanrex/sensors_project/tree/main/model/.sbatch)  for reference.
To replicate our experiments run:

```sh
    user@cluster:~/sensors_project$ sbatch model/.sbatch/<architecture>.sbatch
```

Examples:

```sh
    user@cluster:~/sensors_project$ sbatch model/.sbatch/KNN_cluster_job.sbatch
    user@cluster:~/sensors_project$ sbatch model/.sbatch/LASSO_cluster_job.sbatch
    user@cluster:~/sensors_project$ sbatch model/.sbatch/MLP_cluster_job.sbatch
    user@cluster:~/sensors_project$ sbatch model/.sbatch/RF_cluster_job.sbatch
```

The results of our experiments are stored in the [results](https://github.com/evanrex/sensors_project/tree/main/model/results) folder.
