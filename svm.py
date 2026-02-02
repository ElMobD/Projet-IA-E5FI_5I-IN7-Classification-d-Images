import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ===== Étape 1 : Charger les données =====
print("Chargement des données...")

X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

print(f"X_train shape : {X_train.shape}")
print(f"y_train shape : {y_train.shape}")
print(f"X_test shape : {X_test.shape}")
print(f"y_test shape : {y_test.shape}")

# ===== Étape 2 : Normaliser =====
print("\nNormalisation des données...")

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Normalisation terminée.")

# ===== Étape 3 : Entraîner le SVM =====
print("\nEntraînement du modèle SVM...")

model = SVC(kernel="rbf", C=1.0, gamma="scale")
model.fit(X_train, y_train)

print("Entraînement terminé.")

# ===== Étape 4 : Prédictions =====
print("\nPrédictions sur le test set...")

y_pred = model.predict(X_test)

print("Prédictions terminées.")

# ===== Étape 5 : Évaluation =====
print("\nÉvaluation du modèle...")

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy : {accuracy:.4f}")

cm = confusion_matrix(y_test, y_pred)
print("Matrice de confusion :")
print(cm)

# Affichage graphique
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Dog", "Cat"],
            yticklabels=["Dog", "Cat"])
plt.xlabel("Prédit")
plt.ylabel("Vrai")
plt.title("Matrice de confusion")
plt.show()

# ===== Étape 6 : Sauvegarde du modèle =====
#joblib.dump(model, "svm_model.pkl")
#print("Modèle sauvegardé : svm_model.pkl")