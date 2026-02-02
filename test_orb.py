import cv2
import numpy as np
import matplotlib.pyplot as plt

# Créer le détecteur ORB
orb = cv2.ORB_create()

# IMPORTANT : Remplacez par le chemin de VOTRE image
# Exemples possibles :
# im = cv2.imread("dataset/training_set/dogs/dog.1.jpg")
# im = cv2.imread("mon_image.jpg")
im = cv2.imread("dataset/training_set/dogs/dog.1.jpg")

# Vérifier que l'image est bien chargée
if im is None:
    print("Erreur : Image non trouvée !")
else:
    print(f"Image chargée : {im.shape}")
    

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)

    # Afficher l'image originale
    plt.title("Image originale")
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    # Fonction pour dessiner les keypoints
    def draw_keypoints(vis, keypoints, color=(0, 255, 255)):
        for kp in keypoints:
            x, y = kp.pt
            cv2.circle(vis, (int(x), int(y)), 2, color, -1)
        return vis
    
    # Détecter les keypoints
    kp = orb.detect(im, None)
    print(f"Nombre de keypoints détectés : {len(kp)}")
    
    # Calculer les descripteurs
    kp, des = orb.compute(im, kp)
    print(f"Forme des descripteurs : {des.shape}, {des[0]}")
    
    # Dessiner les keypoints sur l'image
    img_with_keypoints = draw_keypoints(im.copy(), kp)
    
    # Afficher l'image avec les keypoints
    plt.subplot(1, 2, 2)
    plt.title(f"Image avec {len(kp)} keypoints")
    plt.imshow(cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()