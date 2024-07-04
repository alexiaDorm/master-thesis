from models.training_loop import train_w_bias
import torch
import optuna
import joblib

import random 
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

def objective(trial):
    
    model, train_loss, train_KLD, train_MSE, test_KLD, test_MSE, corr_test, jsd_test = train_w_bias(trial, str(trial.number), device, save=True)

    return test_KLD[-1] + 2*test_MSE[-1]

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    joblib.dump(study, "study.pkl")

    #Resume study
    #study = joblib.load("study.pkl")

    study.optimize(objective, n_trials=2)

    pruned_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))