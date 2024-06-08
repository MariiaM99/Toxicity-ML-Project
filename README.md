# Toxicity-Prediction-Project
A machine learning project focused on predicting toxicity using SMILES strings and a variety of classification algorithms.

# Overview
In this project, SMILES strings were preprocessed to generate features, then these features were used to predict toxicity across multiple assays. The prediction models include various machine learning algorithms, such as SVM, Random Forest, AdaBoost, XGBoost, LightGBM, Neural Network (FNN), HistGradientBoostingClassifier. The models were trained on the Tox21_10k dataset.

# Dataset
Tox21_10k
The Tox21_10k dataset is a public dataset. It consists of approximately 10,000 chemical compounds that have been tested for 12 different toxicity endpoints. These endpoints are related to various biological targets and pathways, including nuclear receptor signaling and stress response pathways.

Source: The dataset is part of the Tox21 initiative, a collaboration between several U.S. government agencies, including the National Institutes of Health (NIH), Environmental Protection Agency (EPA), and Food and Drug Administration (FDA).

Data Composition: The dataset includes qualitative results (active/inactive) for each compound across 12 assays. Each compound is represented by ROMol (RDKit molecule object).

Features: The models in this project were trained using molecular descriptors and embeddings derived from the ROMol of the compounds. The molecular descriptors include: molecular weight, lipophilicity, polar surface area, number of heavy atoms, number of heteroatoms, number of hydrogen bond donors, number of hydrogen bond acceptors and number of rotatable bonds. Additionally, mol2vec embeddings were used to capture the structural information of the molecules.

Assays: The 12 toxicity endpoints (assays) included in the dataset are:

SR-HSE: Stress response to heat shock element.

NR-AR: Nuclear receptor signaling for the androgen receptor.

SR-ARE: Stress response to antioxidant response element.

NR-Aromatase: Nuclear receptor signaling for aromatase.

NR-ER-LBD: Nuclear receptor signaling for the estrogen receptor ligand-binding domain.

NR-AhR: Nuclear receptor signaling for the aryl hydrocarbon receptor.

SR-MMP: Stress response to mitochondrial membrane potential.

NR-ER: Nuclear receptor signaling for the estrogen receptor.

NR-PPAR-gamma: Nuclear receptor signaling for the peroxisome proliferator-activated receptor gamma.

SR-p53: Stress response related to the tumor protein p53.

SR-ATAD5: Stress response related to the ATPase family AAA domain-containing protein 5.

NR-AR-LBD: Nuclear receptor signaling for the androgen receptor ligand-binding domain.

# Features
SMILES Preprocessing: Convert SMILES strings into feature vectors. SMILES (Simplified Molecular Input Line Entry System) allows to represent a chemical structure in a way that can be used by various computational programs. SMILES are strings consisted of characters that represent specific atoms, bonds, and connectivity in a molecule.

Molecular Descriptors: Utilize various molecular descriptors (molecular weight, lipophilicity, polar surface area, number of heavy atoms, number of heteroatoms, number of hydrogen bond donors, number of hydrogen bond acceptors, number of rotatable bonds) as features. Molecular descriptors are calculated based on SMILES.

Toxicity Prediction: Predict toxicity for multiple assays (SR-HSE, NR-AR, SR-ARE, NR-Aromatase, NR-ER-LBD, NR-AhR, SR-MMP, NR-ER, NR-PPAR-gamma, SR-p53, SR-ATAD5, NR-AR-LBD) using pre-trained models.

Model Handling: Load and use pre-trained models (SVM, Random Forest, AdaBoost, XGBoost, LightGBM, Neural Network (FNN), HistGradientBoostingClassifier) for prediction.

# Models
The models used in this project include:

Support Vector Machine (SVM)

Random Forest

AdaBoost

XGBoost

LightGBM

Neural Network (Feedforward Neural Network, FNN)

HistGradientBoostingClassifier

# Setup
Prerequisites
Python 3.x

Required Python libraries: rdkit, py3Dmol, mol2vec, gensim, xgboost, lightgbm, pandas, requests, numpy, matplotlib, seaborn, joblib, tensorflow, ipywidgets, scikit-learn, imbalanced-learn

Mol2vec model: https://github.com/samoturk/mol2vec_notebooks/raw/master/Notebooks/model_300dim.pkl

# Deployment
http://85.143.223.223:8503/
