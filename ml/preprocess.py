from tensorflow.keras.preprocessing.image import ImageDataGenerator


def get_data_generators(train_dir, val_dir):
    train_gen = ImageDataGenerator(rescale=1. / 255)
    val_gen = ImageDataGenerator(rescale=1. / 255)

    train_data = train_gen.flow_from_directory(
        train_dir, target_size=(128, 128), batch_size=32, class_mode='categorical')

    val_data = val_gen.flow_from_directory(
        val_dir, target_size=(128, 128), batch_size=32, class_mode='categorical')

    return train_data, val_data
