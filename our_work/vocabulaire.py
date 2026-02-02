import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
from tqdm import tqdm

NBR_CLUSTER = 200  # Nombre de clusters pour K-Means

train_path = "dataset/training_set"
image_paths = []

# Pour les chiens
dogs_path = os.path.join(train_path, "dogs")
for filename in os.listdir(dogs_path):
    image_paths.append(os.path.join(dogs_path, filename))

    # Pour les chats
cats_path = os.path.join(train_path, "cats")
for filename in os.listdir(cats_path):
    image_paths.append(os.path.join(cats_path, filename))

print(f"Nombre total d'images : {len(image_paths)}")

orb = cv2.ORB_create()

des_list = []

print("Extraction des descripteurs...")
for image_path in tqdm(image_paths, desc="Images traitées"):
    # Charger l'image
    img = cv2.imread(image_path)
    
    # Vérifier que l'image est chargée
    if img is None:
        continue
    
    # Détecter keypoints
    kp = orb.detect(img, None)
    
    # Calculer descripteurs
    kp, des = orb.compute(img, kp)
    
    # Ajouter les descripteurs à la liste
    if des is not None:
        des_list.append(des)

print(f"Nombre d'images traitées : {len(des_list)}")

# Empiler tous les descripteurs verticalement
descriptors = des_list[0]  # Commencer avec les descripteurs de la 1ère image

for des in des_list[1:]:
    descriptors = np.vstack((descriptors, des))

descriptors_float = descriptors.astype(float)

k = NBR_CLUSTER  # Nombre de clusters

print("\nCréation du vocabulaire avec K-Means...")
kmeans_model = KMeans(n_clusters=k, verbose=1, n_init=10)
kmeans_model.fit(descriptors_float)
vocabulary = kmeans_model.cluster_centers_

print(f"\nVocabulaire créé ! Forme : {vocabulary.shape}")

# Optionnel car on sais jamais si on va en avoir besoin plus tard
np.save("outputs/vocabulary.npy", vocabulary)
print("Vocabulaire sauvegardé dans outputs/vocabulary.npy")