import cv2
import numpy as np
import os
from scipy.cluster.vq import vq
import random

# Charger le vocabulaire
vocabulary = np.load("vocabulary.npy")
print(f"Vocabulaire chargé : {vocabulary.shape}")

k = 200  # Nombre de mots visuels

# TRAINING SET
train_path = "dataset/training_set"

image_paths_train = []
labels_train = []

# Chiens (classe 0)
for filename in os.listdir(os.path.join(train_path, "dogs")):
    image_paths_train.append(os.path.join(train_path, "dogs", filename))
    labels_train.append(0)

# Chats (classe 1)
for filename in os.listdir(os.path.join(train_path, "cats")):
    image_paths_train.append(os.path.join(train_path, "cats", filename))
    labels_train.append(1)

# TEST SET
test_path = "dataset/test_set"

image_paths_test = []
labels_test = []

# Chiens
for filename in os.listdir(os.path.join(test_path, "dogs")):
    image_paths_test.append(os.path.join(test_path, "dogs", filename))
    labels_test.append(0)

# Chats
for filename in os.listdir(os.path.join(test_path, "cats")):
    image_paths_test.append(os.path.join(test_path, "cats", filename))
    labels_test.append(1)

print(f"\n=== TRAINING SET ===")
print(f"Nombre total d'images : {len(image_paths_train)}")
print(f"Chiens (0) : {labels_train.count(0)}")
print(f"Chats (1) : {labels_train.count(1)}")

print(f"\n=== TEST SET ===")
print(f"Nombre total d'images : {len(image_paths_test)}")
print(f"Chiens (0) : {labels_test.count(0)}")
print(f"Chats (1) : {labels_test.count(1)}")

def image_to_histogram(image_path, orb, vocabulary, k):

    # Charger l'image
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # Détecter keypoints
    kp = orb.detect(img, None)

    # Calculer descripteurs
    kp, des = orb.compute(img, kp)

    # Si l'image n'a pas de descripteurs, retourner un vecteur de zéros
    if des is None:
        return np.zeros(k)
    
    # Assigner chaque descripteur au mot visuel le plus proche
    words, distances = vq(des, vocabulary)

    # Créer l'histogramme (compter les occurrences de chaque mot)
    histogram = np.zeros(k)
    for word in words:
        histogram[word] += 1
    
    return histogram

orb = cv2.ORB_create()

print("\n--- Transformation du TRAINING SET ---")
X_train = []  # X_train = histogrammes des images d'entraînement (features)
y_train = []  # y_train = labels des images (0=chien, 1=chat)

for i, (image_path, label) in enumerate(zip(image_paths_train, labels_train)):
    # Afficher la progression tous les 500 images
    if i % 500 == 0:
        print(f"Progression : {i}/{len(image_paths_train)}")
    
    # Transformer l'image en histogramme
    histogram = image_to_histogram(image_path, orb, vocabulary, k)
    
    # Si l'image a pu être traitée
    if histogram is not None:
        X_train.append(histogram)
        y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

print(f"X_train shape : {X_train.shape}")
print(f"y_train shape : {y_train.shape}")

print("\n--- Transformation du TEST SET ---")
X_test = []
y_test = []

for i, (image_path, label) in enumerate(zip(image_paths_test, labels_test)):
    if i % 100 == 0:
        print(f"Progression : {i}/{len(image_paths_test)}")
    
    histogram = image_to_histogram(image_path, orb, vocabulary, k)
    
    if histogram is not None:
        X_test.append(histogram)
        y_test.append(label)

X_test = np.array(X_test)
y_test = np.array(y_test)

print(f"X_test shape : {X_test.shape}")
print(f"y_test shape : {y_test.shape}")


print("\n--- Sauvegarde des données ---")

np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)
np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)

print("✓ X_train.npy sauvegardé")
print("✓ y_train.npy sauvegardé")
print("✓ X_test.npy sauvegardé")
print("✓ y_test.npy sauvegardé")

print("\n=== PHASE 3 TERMINÉE ===")
print(f"X_train : {X_train.shape}")
print(f"X_test : {X_test.shape}")