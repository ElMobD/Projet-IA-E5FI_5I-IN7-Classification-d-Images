import cv2
import numpy as np
import os
from scipy.cluster.vq import kmeans

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

for image_path in image_paths:
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

k = 200 # Nombre de clusters

print("Création du vocabulaire avec K-Means... (cela peut prendre du temps)")
vocabulary, variance = kmeans(descriptors_float, k, 1)

print(f"Vocabulaire créé ! Forme : {vocabulary.shape}")
print(f"Variance : {variance}")

#Optionnel car on sais jamais si on va en avoir besoin plus tard
np.save("vocabulary.npy", vocabulary)
print("Vocabulaire sauvegardé dans vocabulary.npy")