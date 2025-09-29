import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, image
from tensorflow.keras import layers, models

IMG_HEIGHT, IMG_WIDTH = 150, 150
BATCH_SIZE = 32
EPOCHS = 10

DATASET_PATH = "dataset/Wound_dataset"   # update path to your dataset
TEST_IMAGE_PATH = "dataset/sample.jpg"   # update with test image path
TREATMENT_CSV = "dataset/wound_treatment_5step.csv"

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

validation_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(len(train_generator.class_indices), activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS
)

model.save("wound_classification_model.h5")

img = image.load_img(TEST_IMAGE_PATH, target_size=(IMG_HEIGHT, IMG_WIDTH))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

predictions = model.predict(img_array)
predicted_class = np.argmax(predictions)
class_labels = {v: k for k, v in train_generator.class_indices.items()}
predicted_label = class_labels[predicted_class]

print(f"Predicted Wound Type: {predicted_label}")

treatment_data = pd.read_csv(TREATMENT_CSV)

def recommend_treatment(predicted_label):
    treatment_row = treatment_data[treatment_data["Wound_Type"] == predicted_label]
    if not treatment_row.empty:
        return treatment_row["Treatment"].values[0]
    else:
        return "Consult a healthcare professional for advice."

treatment = recommend_treatment(predicted_label)
print(f"Treatment Recommendation: {treatment}")