import pandas as pd
import numpy as np

# CSV avec separateur espace
df = pd.read_csv("data/patients_dakar.csv", sep='\s+', engine='python')

# Nettoyer les noms de colonnes
df.columns = df.columns.str.strip()

# Verifier les dimensions
print(f"Dataset : {df.shape[0]} patients, {df.shape[1]} colonnes")
print(f"\nColonnes : {list(df.columns)}")
print(f"\nDiagnostics :\n{df['diagnostic'].value_counts()}")

from sklearn.preprocessing import LabelEncoder

# Encoder les variables categoriques
le_sexe = LabelEncoder()
le_region = LabelEncoder()

df['sexe_encoded'] = le_sexe.fit_transform(df['sexe'])
df['region_encoded'] = le_region.fit_transform(df['region'])

# Definir les features (X) et la cible (y)
feature_cols = ['age', 'sexe_encoded', 'temperature', 'tension_sys',
                'toux', 'fatigue', 'maux_tete', 'region_encoded']

X = df[feature_cols]
y = df['diagnostic']

print(f"Features : {X.shape}")
print(f"Cible : {y.shape}")

from sklearn.model_selection import train_test_split

# 80% pour l'entrainement, 20% pour le test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 20% pour le test
    random_state=42,     # Pour avoir les memes resultats
    stratify=y           # Garder les memes proportions
)

print(f"Entrainement : {X_train.shape[0]} patients")
print(f"Test : {X_test.shape[0]} patients")

from sklearn.ensemble import RandomForestClassifier

# Creer le modele
model = RandomForestClassifier(
    n_estimators=100,  # 100 arbres de decision
    random_state=42    # Reproductibilite
)

# Entrainer sur les donnees d'entrainement
model.fit(X_train, y_train)

print("Modele entraine !")
print(f"Nombre d'arbres : {model.n_estimators}")
print(f"Nombre de features : {model.n_features_in_}")
print(f"Classes : {list(model.classes_)}")

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Predire sur les donnees de test
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy : {accuracy:.2%}")

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
print("\nMatrice de confusion :")
print(cm)

# Rapport de classification
print("\nRapport de classification :")
print(classification_report(y_test, y_pred))

# Visualiser avec seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=model.classes_,
            yticklabels=model.classes_)
plt.xlabel('Prediction du modele')
plt.ylabel('Vrai diagnostic')
plt.title('Matrice de confusion - SenSante')
plt.tight_layout()
plt.savefig('figures/confusion_matrix.png', dpi=150)
plt.show()
print("Figure sauvegardee !")

import joblib
import os

# ============================================
# ETAPE 6 : Serialiser le modele
# ============================================

# Creer le dossier models/ s'il n'existe pas
os.makedirs("models", exist_ok=True)

# Serialiser le modele
joblib.dump(model, "models/model.pkl")

# Verifier la taille du fichier
size = os.path.getsize("models/model.pkl")
print(f"Modele sauvegarde : models/model.pkl")
print(f"Taille : {size / 1024:.1f} Ko")

# Sauvegarder aussi les encodeurs
joblib.dump(le_sexe, "models/encoder_sexe.pkl")
joblib.dump(le_region, "models/encoder_region.pkl")
joblib.dump(feature_cols, "models/feature_cols.pkl")

print("Encodeurs et metadata sauvegardes.")

# ============================================
# ETAPE 7 : Tester le modele serialise
# ============================================

model_loaded = joblib.load("models/model.pkl")
le_sexe_loaded = joblib.load("models/encoder_sexe.pkl")
le_region_loaded = joblib.load("models/encoder_region.pkl")
feature_cols_loaded = joblib.load("models/feature_cols.pkl")

print(f"Modele recharge : {type(model_loaded).__name__}")
print(f"Classes : {list(model_loaded.classes_)}")

# Nouveau patient
nouveau_patient = {
    'age': 28,
    'sexe': 'F',
    'temperature': 39.5,
    'tension_sys': 110,
    'toux': 1,
    'fatigue': 1,
    'maux_tete': 1,
    'frissons': 1,
    'nausee': 0,
    'region': 'Dakar'
}

# Encoder
sexe_enc = le_sexe_loaded.transform([nouveau_patient['sexe']])[0]
region_enc = le_region_loaded.transform([nouveau_patient['region']])[0]

patient_encoded = {
    'age': nouveau_patient['age'],
    'sexe_encoded': sexe_enc,
    'temperature': nouveau_patient['temperature'],
    'tension_sys': nouveau_patient['tension_sys'],
    'toux': nouveau_patient['toux'],
    'fatigue': nouveau_patient['fatigue'],
    'maux_tete': nouveau_patient['maux_tete'],
    'frissons': nouveau_patient['frissons'],
    'nausee': nouveau_patient['nausee'],
    'region_encoded': region_enc
}

features = [patient_encoded[col] for col in feature_cols_loaded]

# Predire
diagnostic = model_loaded.predict([features])[0]
probas = model_loaded.predict_proba([features])[0]
proba_max = probas.max()

print(f"\n--- Resultat du pre-diagnostic ---")
print(f"Patient : {nouveau_patient['sexe']}, {nouveau_patient['age']} ans")
print(f"Diagnostic : {diagnostic}")
print(f"Probabilite : {proba_max:.1%}")
print(f"\nProbabilites par classe :")
for classe, proba in zip(model_loaded.classes_, probas):
    bar = '#' * int(proba * 30)
    print(f"  {classe:12s} : {proba:.1%} {bar}")