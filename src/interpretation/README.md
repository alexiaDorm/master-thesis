Quick description of scripts
--------------------------------------------------------------

- interpret.py: Functions definition for interpretation of the model (compute shap's Deeplift implementation and attribution maps visualization)
- overwrite_shap_explainer.py: Overwrite definition of shap's class: DeepExplainer to make it compatible with model outputs format
- synthetic_seq_analysis.py: Generate random DNA sequences and motifs from frequency matrix of TF motifs and insert randomly inside generated synthetic sequences