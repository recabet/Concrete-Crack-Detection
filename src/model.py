import tensorflow as tf
from typing import Optional, Tuple, List


def simple_unet_model(input_size: Tuple[int, int, int] = (128, 128, 3)) -> tf.keras.models.Model:
    """
    Builds a simple U-Net model.

    Parameters:
    - input_size: Tuple[int, int, int], the size of the input image (default is (128, 128, 3))

    Returns:
    - model: tf.keras.models.Model, the compiled U-Net model
    """
    inputs = tf.keras.layers.Input(input_size)
    conv1 = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(inputs)
    conv1 = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(conv1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    up2 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(pool1)
    up2 = tf.keras.layers.concatenate([up2, conv1])
    conv2 = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(up2)
    conv2 = tf.keras.layers.Conv2D(1, 1, activation="sigmoid")(conv2)
    model = tf.keras.models.Model(inputs, conv2)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", tf.keras.metrics.BinaryIoU()])

    return model


def complex_unet_model(input_size: Tuple[int, int, int] = (256, 256, 1)) -> tf.keras.models.Model:
    """
    Builds a more complex U-Net model with deeper layers.

    Parameters:
    - input_size: Tuple[int, int, int], the size of the input image (default is (256, 256, 1))

    Returns:
    - model: tf.keras.models.Model, the compiled U-Net model
    """
    inputs = tf.keras.layers.Input(input_size)
    c1 = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    c1 = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(p1)
    c2 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

    c3 = tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same")(p2)
    c3 = tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same")(c3)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

    c4 = tf.keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same")(p3)
    c4 = tf.keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same")(c4)
    p4 = tf.keras.layers.MaxPooling2D((2, 2))(c4)

    c5 = tf.keras.layers.Conv2D(512, (3, 3), activation="relu", padding="same")(p4)
    c5 = tf.keras.layers.Conv2D(512, (3, 3), activation="relu", padding="same")(c5)

    u6 = tf.keras.layers.UpSampling2D((2, 2))(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    c6 = tf.keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same")(u6)
    c6 = tf.keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same")(c6)

    u7 = tf.keras.layers.UpSampling2D((2, 2))(c6)
    u7 = tf.keras.layers.concatenate([u7, c3])
    c7 = tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same")(u7)
    c7 = tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same")(c7)

    u8 = tf.keras.layers.UpSampling2D((2, 2))(c7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    c8 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(u8)
    c8 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(c8)

    u9 = tf.keras.layers.UpSampling2D((2, 2))(c8)
    u9 = tf.keras.layers.concatenate([u9, c1])
    c9 = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(u9)
    c9 = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(c9)

    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation="relu")(c9)

    model = tf.keras.models.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", tf.keras.metrics.BinaryIoU()])

    return model


def create_hist(
    model: tf.keras.models.Model,
    X_train: tf.Tensor,
    y_train: tf.Tensor,
    X_test: tf.Tensor,
    y_test: tf.Tensor,
    epochs: int,
    batch_size: int,
    verbose: bool,
    validation_split: float = 0.0,
    callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
    datagen: Optional[tf.keras.preprocessing.image.ImageDataGenerator] = None
) -> tf.keras.callbacks.History:
    """
    Trains the model and returns the training history.

    Parameters:
    - model: tf.keras.models.Model, the model to train
    - X_train: tf.Tensor, the training input data
    - y_train: tf.Tensor, the training labels
    - X_test: tf.Tensor, the test input data
    - y_test: tf.Tensor, the test labels
    - epochs: int, number of training epochs
    - batch_size: int, the batch size
    - verbose: bool, verbosity mode (True for progress bar, False for silent)
    - validation_split: float, fraction of training data used as validation set (default is 0.0)
    - callbacks: Optional[List[tf.keras.callbacks.Callback]], list of callbacks (default is None)
    - datagen: Optional[tf.keras.preprocessing.image.ImageDataGenerator], data generator (default is None)

    Returns:
    - history: tf.keras.callbacks.History, the training history object
    """
    if datagen:
        if validation_split == 0.0:
            return model.fit(
                datagen.flow(X_train, y_train, batch_size=batch_size),
                epochs=epochs,
                validation_data=(X_test, y_test),
                callbacks=callbacks,
                verbose=verbose,
            )
        elif validation_split > 0.0:
            return model.fit(
                datagen.flow(X_train, y_train, batch_size=batch_size),
                epochs=epochs,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=verbose,
            )
    else:
        if validation_split == 0.0:
            return model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_test, y_test),
                callbacks=callbacks,
                verbose=verbose
            )
        else:
            return model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=verbose
            )
