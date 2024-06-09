
# streamlit run mol.py --server.runOnSave True

# pip install rdkit
# pip install mol2vec gensim
# pip install scikit-learn
# pip install imblearn
      
import os          
import numpy as np
import pandas as pd
import joblib
import streamlit as st


from rdkit import Chem
from rdkit.Chem import PandasTools
from rdkit.Chem import Draw

from rdkit.Chem import Crippen
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Descriptors
from rdkit.Chem import MolFromSmiles

from mol2vec.features import mol2alt_sentence, MolSentence
import gensim
import pickle
from mol2vec.features import mol2sentence, DfVec, sentences2vec
from mol2vec.helpers import depict_identifier, plot_2D_vectors, IdentifierTable, mol_to_svg
from gensim.models import Word2Vec


def fn_read(file_name: str) -> str:
    with open (file_name, 'r', encoding='utf-8') as f:
        return f.read()

def sentences2vec(sentences, model, unseen=None):
    keys = set(model.wv.key_to_index)
    vec = []
    for sentence in sentences:
        this_vec = []
        for word in sentence:
            if word in keys:
                this_vec.append(model.wv[word])
            elif unseen:
                this_vec.append(model.wv[unseen])
        if this_vec:
            vec.append(np.mean(this_vec, axis=0))
        else:
            vec.append(np.zeros(model.vector_size))
    return vec

def preprocess_smiles(smiles, model):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Calculating descriptors
    descriptors = [
        Descriptors.MolLogP(mol),
        Descriptors.MolWt(mol),
        Descriptors.TPSA(mol),
        Descriptors.HeavyAtomCount(mol),
        Descriptors.NumHeteroatoms(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumRotatableBonds(mol)
    ]

    # Converting to MolSentence
    sentence = MolSentence(mol2alt_sentence(mol, 1))

    # Converting sentence to vector using sentences2vec function
    vec = sentences2vec([sentence], model, unseen='UNK')[0]

    # Combining descriptors and mol2vec vector into a single feature array
    features = np.concatenate([descriptors, vec])

    return features

def predict_models(features, models):
    predictions = {}
    for col, model in models.items():
        prediction = model.predict(features)
        predictions[col] = prediction[0][0]
    return predictions


# Placeholder for target columns
target_columns = ['SR-HSE', 'NR-AR', 'SR-ARE', 'NR-Aromatase', 'NR-ER-LBD', 'NR-AhR', 'SR-MMP', 'NR-ER', 'NR-PPAR-gamma', 'SR-p53', 'SR-ATAD5', 'NR-AR-LBD']

# Directory where models are saved
save_dir = "/root/mol/model_6"

print_code = st.sidebar.checkbox("Print code")


st.markdown("<h1>Toxicity Prediction</h1>", unsafe_allow_html=True)

st.markdown("Predictive model of toxicity of chemical compounds using RD-Kit/Word2Vec/scikit-learn", unsafe_allow_html=True)


st.subheader('Formula')

smiles_ = 'C/C=C/C=C/C(O)=O'

smiles = st.text_input("SMILES", smiles_)

st.write("The current formula:", smiles)

model = Word2Vec.load('model_300dim.pkl')

# Preprocessing the SMILES string
features = preprocess_smiles(smiles, model)

#st.subheader('Features')
#st.write(features)

st.subheader('Predictions')

if features is None:
    print("Invalid SMILES string.")
    st.write("Invalid SMILES string.")

else:
    # Ensuring the feature array has the correct shape
    features = features.reshape(1, -1)

    # Making predictions using all models
    for col in target_columns:
        model_path_6 = os.path.join(save_dir, f"model_6_{col}.pkl")
        model_6 = joblib.load(model_path_6)
        y_pred = model_6.predict_proba(features)
        prediction = y_pred[0][1]
        st.write(f"Prediction for {col}: {prediction:.4f}")
        #st.write(y_pred)
        #st.write(prediction)


if print_code:
    st.subheader('Code')
    code = fn_read('mol.py')
    st.code(code, language="python", line_numbers=True)
