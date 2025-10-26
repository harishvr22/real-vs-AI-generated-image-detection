import os
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# CONFIG - change paths if needed
DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 25

MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), "saved_model", "model.h5")
CLASSES_JSON = os.path.join(os.path.dirname(__file__), "saved_model", "classes.json")

def build_model(input_shape=(128,128,3)):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    # Use ImageDataGenerator with validation split
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2,
                                 horizontal_flip=True, rotation_range=10, zoom_range=0.1)
    train_gen = datagen.flow_from_directory(DATASET_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary', subset='training')
    val_gen = datagen.flow_from_directory(DATASET_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary', subset='validation')

    # Save class indices mapping
    class_indices = train_gen.class_indices
    inv_map = {v:k for k,v in class_indices.items()}
    os.makedirs(os.path.dirname(CLASSES_JSON), exist_ok=True)
    with open(CLASSES_JSON, 'w') as f:
        json.dump(inv_map, f)
    print("Saved class mapping to", CLASSES_JSON)

    model = build_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    print(model.summary())

    callbacks = [
        ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_loss', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
    ]

    model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=callbacks)
    model.save(MODEL_SAVE_PATH)
    print("Model saved to", MODEL_SAVE_PATH)

if __name__ == '__main__':
    main()
