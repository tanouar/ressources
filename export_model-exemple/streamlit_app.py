# Importer les bibliothèques nécessaires
import streamlit as st
import pickle
import json
import numpy as np

def charger_modele():
    # Charger le modèle à partir du fichier Pickle
    with open('modele.pkl', 'rb') as fichier_modele:
        modele = pickle.load(fichier_modele)
    return modele

def charger_min_max():
    # Charger les valeurs min et max des caractéristiques depuis le fichier JSON
    with open('feature_min_max.json', 'r') as json_file:
        min_max_dict = json.load(json_file)
    return min_max_dict

def charger_target_mapping():
    # Charger le mapping des targets depuis le fichier JSON
    with open('target_encoding.json', 'r') as json_file:
        target_mapping = json.load(json_file)
    # Convertir les clés en entiers
    target_mapping = {int(key): value for key, value in target_mapping.items()}
    return target_mapping

# Charger les valeurs min et max
min_max_dict = charger_min_max()

# Interface utilisateur Streamlit
st.title("Application de Classification des espèces de fleurs")

# Créer des curseurs pour chaque caractéristique en utilisant les noms et valeurs depuis le JSON
caracteristiques_entree = []
for feature, limits in min_max_dict.items():
    caracteristique = st.slider(
        f"{feature}", 
        float(limits['min']), 
        float(limits['max']), 
        float((limits['min'] + limits['max']) / 2)
    )
    caracteristiques_entree.append(caracteristique)

# Charger le modèle et le mapping de la cible
modele = charger_modele()
target_mapping = charger_target_mapping()

# Préparer les caractéristiques pour la prédiction
caracteristiques = np.array([caracteristiques_entree])

# Prévoir la classe avec le modèle
prediction_encoded = modele.predict(caracteristiques)

# Décoder la prédiction
prediction_decoded = target_mapping[prediction_encoded[0]]

# Afficher la prédiction
st.markdown(
    f"<p style='font-size:24px; font-weight:bold;'>La prédiction de l'espèce est : {prediction_decoded}</p>", 
    unsafe_allow_html=True
)
