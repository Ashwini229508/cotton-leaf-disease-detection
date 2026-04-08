import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

def load_data(train_dir, val_dir, test_dir):

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True
    )

    val_test_datagen = ImageDataGenerator(rescale=1./255)

    train_data = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    val_data = val_test_datagen.flow_from_directory(
        val_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    test_data = val_test_datagen.flow_from_directory(
        test_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    print("\n===== DATASET DETAILS =====")
    print("Image Size:", IMG_SIZE)
    print("Batch Size:", BATCH_SIZE)
    print("Classes:", train_data.class_indices)
    print("Training Images:", train_data.samples)
    print("Validation Images:", val_data.samples)
    print("Test Images:", test_data.samples)

    return train_data, val_data, test_data


if __name__ == "__main__":
    train_dir = "train"
    val_dir = "val"
    test_dir = "test"

    load_data(train_dir, val_dir, test_dir)