# Git_Master_Dissertation

An Interpretable Pipeline for Imbalanced Industrial Anomaly Detection
VAE-GAN Augmentation · CatBoost Classification · TreeSHAP Interpretation
=======================================================================

This project implements an end-to-end anomaly-detection system for high-dimensional industrial sensor data.  
It combines:  
    • VAE-GAN for synthetic-anomaly generation.  
    • CatBoost for classification.  
    • TreeSHAP for model interpretability.  
    • A FastAPI server for real-time streaming simulation.

-----------------------------------------------------------------------
1. DATA PREPROCESSING
-----------------------------------------------------------------------
- Input file: 'uci-secom.csv'   (590-feature semiconductor dataset)
- Run 'EDA.py' to
    * impute missing values (mean).
    * remove low-variance features.
    * drop features weakly correlated with the label.
    * keep one feature within highly-correlated pairs.

-----------------------------------------------------------------------
2. USAGE
-----------------------------------------------------------------------
2.1  Generate anomalous samples.  

    Python 'VAE.py' demonstrates the architecture of VAE ; 
    Python 'VAEGAN.py' demonstrates the architecture of VAE-GAN. 
    
    Run python 'GeneratorVAE.py'     # baseline VAE
    Run python 'GeneratorVAEGAN.py'  # improved VAE-GAN

2.2  Evaluate sample quality (KL, JS, Wasserstein, MMD).

    Run python 'Measure.py'

2.3  Train & optimize CatBoost (Bayesian search + early stopping).

    Run python 'CatShap.py'
    Outputs: precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix, ROC & PR curves

2.4  SHAP interpretation: 

      • Bar plot – global feature importance
      • Beeswarm plot – value vs SHAP distribution
      • Waterfall plot – instance-level explanation

2.5  Real-time streaming data validation (10 samples: 7 normal, 3 anomalous).

    Run python 'API.py'              # starts FastAPI on port 8000
    Outputs: {"score": float, "is_anomaly": bool, "plot_file": str | null}

-----------------------------------------------------------------------
3. RESULTS (UCI-SECOM)
-----------------------------------------------------------------------
| Model | F1-score |
|-------|----------|
| VAE + CatBoost | 0.965 |
| **VAE-GAN + CatBoost (best)** | **0.971** |

    • VAE-GAN samples are closer to real anomalies compared with VAE-generated samples.
    • SHAP ranks Feature 511 as the most influential.
    • Beeswarm and waterfall plots provide global and instance-level explainations.
