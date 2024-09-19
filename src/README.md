Quick description of scripts
--------------------------------------------------------------

1. exploration_peaks.ipynb: Analysis of accessible region (number per time point, overlap between time points, ...)

2. load_pytorch_dataset.py: Load the pytorch dataset with all the training examples and store them in pickle file for fast access.

3. Training of the model:
    - train_w_bias.py: Training of model with bias correction.
    - train_wo_bias.py: Training of model without bias correction
    - optimize_hyperparameters.py: Run bayesian optimization for learning hyperparameters using optuna

    - compare_models.py: Visualise and compare performance of different models

4. Validation of learnt representation:
    - tn5_bias_check.py: Check if model recognized tn5 bias and considers it important for its predictions
    - visu_first_filter.ipynb: Determine the consensus sequence activating each filter and matched them to know TF motifs. The results can be found in results/tomtom_out/
    - TF_check.py: Check if model can regognize the provided TF motifs
    - motif_discovery.ipynb: Find TF motifs in attribution maps of test sequences using DeepLift/TF-MOdisco 

    - predict.py: Generate predictions for each sequences from the training data
    - inspect_pred.ipynb: Look at first layer cell type encoding weights and compare prediction to observe accessibility.

 5. Estimation of variants effect on chromatin accessibility
    - predict_reg_regions.py: Make predictions for accessible regions with and without the variants found in accessible regulatory regions. Identify the variants disturbing chromatin accessibility and compute attribution maps for them.
    - estimate_var_effect.ipynb: Inspect predictions of reference and mutated alleles and prioritised variants. Inspect prioritized attributions maps around varaints


