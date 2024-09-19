Quick description of scripts
--------------------------------------------------------------

1. Definition dataset and models classes
    
    - models.py: define the bias and main pytorch models architechture
    - pytorch_datasets.py: pytorch datasets class storing sequence/accessibility pairs

2. Evaluation

    - eval_metrics.py: definition of the loss and validation evalution metrics for profile and counts

3. Training 
    - training_loop.py: Training loop for training the model with bias correction. Intended to be used during hyperparameters search by oputna.