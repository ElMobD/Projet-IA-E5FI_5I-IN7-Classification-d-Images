import numpy as np
import cv2
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# Configuration
# ============================================================================
print("="*60)
print("Deep Learning - CNN")
print("="*60)

print("\n[1/6] Chargement des images d'entraînement...")
""" Première méthode utilisé pour le chargement des images sans ImageDataGenerator
images_train = []
labels_train = []

cats_path = "./dataset/training_set/cats"
for img_name in os.listdir(cats_path):
    img = cv2.imread(os.path.join(cats_path, img_name))
    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    images_train.append(img)
    labels_train.append(0)

dogs_path = "./dataset/training_set/dogs"
for img_name in os.listdir(dogs_path):
    img = cv2.imread(os.path.join(dogs_path, img_name))
    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    images_train.append(img)
    labels_train.append(1)

X_train = np.array(images_train)
Y_train = np.array(labels_train)
print(f"✓ X_train shape: {X_train.shape}")
print(f"✓ y_train shape: {Y_train.shape}")

print("\n[2/6] Chargement des images de test...")

images_test = []
labels_test = []

cats_test_path = "./dataset/test_set/cats"
for img_name in os.listdir(cats_test_path):
    img = cv2.imread(os.path.join(cats_test_path, img_name))
    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    images_test.append(img)
    labels_test.append(0)

dogs_test_path = "./dataset/test_set/dogs"
for img_name in os.listdir(dogs_test_path):
    img = cv2.imread(os.path.join(dogs_test_path, img_name))
    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    images_test.append(img)
    labels_test.append(1)

X_test = np.array(images_test)
Y_test = np.array(labels_test)
print(f"✓ X_test shape: {X_test.shape}")
print(f"✓ y_test shape: {Y_test.shape}")
"""
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Générateur pour l'entraînement AVEC augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Chargement des données
training_set = train_datagen.flow_from_directory(
    './dataset/training_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary',
    shuffle=True
)
print(f"✓ Training samples: {training_set.samples}")

print("\n[2/6] Chargement des images de test...")

# Générateur pour le test SANS augmentation
test_datagen = ImageDataGenerator(rescale=1./255)

test_set = test_datagen.flow_from_directory(
    './dataset/test_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

print("\n[3/6] Construction du modèle CNN...")

model = Sequential()

# Bloc 1 : Conv + MaxPooling
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D((2, 2)))

# Bloc 2 : Conv + MaxPooling
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# Bloc 3 : Conv + MaxPooling
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# Bloc 4 : Conv + MaxPooling (nouvelle couche)
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# Flatten : Transformation en vecteur
model.add(Flatten())

# Dense : Couches fully-connected
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.summary()
print("✓ Modèle construit")

print("\n[4/6] Compilation du modèle...")

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

print("✓ Modèle compilé")

print("\n[5/6] Entraînement du modèle...")

""" Première méthode utilisé pour l'entrainement sans validation split
history = model.fit(
    X_train, Y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)
"""
history = model.fit(
    training_set,
    steps_per_epoch=training_set.samples // 32,
    epochs=25,
    validation_data=test_set,
    validation_steps=test_set.samples // 32,
    verbose=1
)

print("✓ Entraînement terminé")

print("\n[6/6] Évaluation sur le test set...")

test_set.reset()
y_pred_proba = model.predict(test_set, steps=test_set.samples // 32 + 1)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()
y_true = test_set.classes[:len(y_pred)]
accuracy = accuracy_score(y_true, y_pred)

print(f"\n{'='*60}")
print(f"ACCURACY CNN : {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"{'='*60}")

# Matrice de confusion
cm = confusion_matrix(y_true, y_pred)
print("\nMatrice de Confusion:")
print(cm)

# Classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=['Cats', 'Dogs']))