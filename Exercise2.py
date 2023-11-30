import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf

# os.chdir("E:/")

masked_dir = 'Dataset/with_mask'
unmasked_dir = 'Dataset/without_mask'

# extract features from images
def extract_features(directory, label):
    features = []
    labels = []
    
    for filename in os.listdir(directory):
        img_path = os.path.join(directory, filename)
        img = image.load_img(img_path, target_size=(150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize pixel values
        
        features.append(img_array)
        labels.append(label)
    
    return np.vstack(features), labels

# for masked images
masked_features, masked_labels = extract_features(masked_dir, 1)

#  for unmasked images
unmasked_features, unmasked_labels = extract_features(unmasked_dir, 0)

# Concat features and labels
all_features = np.vstack([masked_features, unmasked_features])
all_labels = np.array(masked_labels + unmasked_labels)

# Reshape to 1D
all_features_1d = all_features.reshape((all_features.shape[0], -1))

# Shuffle 
all_features_shuffled, all_labels_shuffled = shuffle(all_features_1d, all_labels, random_state=42)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    all_features_shuffled, all_labels_shuffled, test_size=0.2, random_state=42
)

# Reshape to 2D for CNN input
X_train = X_train.reshape((X_train.shape[0], 150, 150, 3))
X_test = X_test.reshape((X_test.shape[0], 150, 150, 3))


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Use 'sigmoid' for binary classification
])

model.summary()

# Compile 
model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.RMSprop(lr=1e-4),
              metrics=['accuracy'])

# Train 
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

# Evaluate 
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc*100:.2f}%")

model.save("ninety_eight.h5") # 99.2%


""" PREDICTION"""
from keras.models import load_model

best_model = load_model("ninety_eight.h5")


import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import numpy as np

def load_and_predict(model, img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize pixel values

    # Display the original image
    plt.imshow(img)
    plt.axis('off')
    plt.show()

    # Predict
    prediction = model.predict(img_array)

    # Assigning label based on the prediction
    if prediction[0][0] > 0.5:
        label = 'Masked'
    else:
        label = 'Unmasked'

 
    plt.imshow(img)
    plt.text(10, 10, f'Predicted: {label}', color='red', fontsize=12, weight='bold')
    plt.axis('off')
    plt.show()

image_path_unmasked = 'Dataset/without_mask/tom.jpeg'

image_path_masked = 'Dataset/with_mask/a.jpeg'

load_and_predict(best_model, image_path_unmasked)
load_and_predict(best_model, image_path_masked)











c













""" SAVING TO CSV"""

import os
import pandas as pd
from tensorflow.keras.preprocessing import image
import numpy as np

# Define paths
masked_dir = 'Dataset/with_mask'
unmasked_dir = 'Dataset/without_mask'

# Function to extract features from images
def extract_features(directory, label):
    features = []
    labels = []
    
    for filename in os.listdir(directory):
        img_path = os.path.join(directory, filename)
        img = image.load_img(img_path, target_size=(150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize pixel values
        
        features.append(img_array)
        labels.append(label)
    
    return np.vstack(features), labels

# Extract features for masked images
masked_features, masked_labels = extract_features(masked_dir, 1)

# Extract features for unmasked images
unmasked_features, unmasked_labels = extract_features(unmasked_dir, 0)

# Concatenate features and labels
all_features = np.vstack([masked_features, unmasked_features])
all_labels = masked_labels + unmasked_labels

# Reshape features to 1D
all_features_1d = all_features.reshape((all_features.shape[0], -1))

# Save to CSV
data = {'label': all_labels}
for i in range(all_features_1d.shape[1]):
    data[f'pixel_{i}'] = all_features_1d[:, i]

df = pd.DataFrame(data)
df.to_csv('image_features.csv', index=False)







""" -------------"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define data paths
train_dir = 'Dataset'
test_dir = 'Dataset'

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.RMSprop(lr=1e-4),
              metrics=['accuracy'])

history = model.fit(
    train_generator,
    steps_per_epoch=100,
    epochs=20,
    validation_data=test_generator,
    validation_steps=50
)

test_loss, test_acc = model.evaluate(test_generator)
print(f"Test accuracy: {test_acc*100:.2f}%")