from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np


def display_image_and_mask (image: np.ndarray, mask: np.ndarray) -> None:
    """
    Displays an image and its corresponding mask side by side.

    Args:
        image (np.ndarray): The original image to display.
        mask (np.ndarray): The mask corresponding to the image to display.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image)
    axes[0].set_title("Image")
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Mask")
    plt.show()


def display_test_prediction (X_test: np.ndarray, y_test: np.ndarray, predictions: np.ndarray, num: int = 3) -> None:
    """
    Displays a comparison between original images, true masks, and predicted masks for a test set.

    Args:
        X_test (np.ndarray): The original test images.
        y_test (np.ndarray): The true masks for the test images.
        predictions (np.ndarray): The predicted masks from the model.
        num (int, optional): The number of images to display. Defaults to 3.
    """
    for i in range(num):
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(X_test[i])
        plt.title("Original Image")
        
        plt.subplot(1, 3, 2)
        plt.imshow(y_test[i].reshape(128,128),cmap="gray")
        plt.title("True Mask")
        
        plt.subplot(1, 3, 3)
        plt.imshow(predictions[i].reshape(128,128),cmap="gray")
        plt.title("Predicted Mask")
        
        plt.show()

def plot_training_history (history: tf.keras.callbacks.History) -> None:
    """
    Plots the training and validation metrics (Loss, IoU, Accuracy, Precision, Recall)
    from a Keras History object after model training.

    Parameters:
    ----------
    history : History
        Keras History object returned by the model.fit() method, which contains the metrics
        collected during the training and validation phases.

    The following metrics are expected to be present in the history object:
    - loss
    - val_loss
    - binary_io_u
    - val_binary_io_u
    - accuracy
    - val_accuracy
    - precision
    - val_precision
    - recall
    - val_recall
    """
    
    loss = history.history.get("loss")
    val_loss = history.history.get("val_loss")
    
    iou = history.history.get("binary_io_u")
    val_iou = history.history.get("val_binary_io_u")
    
    acc = history.history.get("accuracy")
    val_acc = history.history.get("val_accuracy")
    
    precision = history.history.get("precision")
    val_precision = history.history.get("val_precision")
    
    recall = history.history.get("recall")
    val_recall = history.history.get("val_recall")
    
    plt.figure(figsize=(12, 20))
    
    plt.subplot(5, 1, 1)
    if loss is not None and val_loss is not None:
        plt.plot(history.epoch, loss, label="Training Loss")
        plt.plot(history.epoch, val_loss, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()

    if iou is not None and val_iou is not None:
        plt.subplot(5, 1, 2)
        plt.plot(history.epoch, iou, label="Training IoU")
        plt.plot(history.epoch, val_iou, label="Validation IoU")
        plt.xlabel("Epochs")
        plt.ylabel("IoU")
        plt.title("Training and Validation IoU")
        plt.legend()
    
    if acc is not None and val_acc is not None:
        plt.subplot(5, 1, 3)
        plt.plot(history.epoch, acc, label="Training Accuracy")
        plt.plot(history.epoch, val_acc, label="Validation Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Training and Validation Accuracy")
        plt.legend()
    
    if precision is not None and val_precision is not None:
        plt.subplot(5, 1, 4)
        plt.plot(history.epoch, precision, label="Training Precision")
        plt.plot(history.epoch, val_precision, label="Validation Precision")
        plt.xlabel("Epochs")
        plt.ylabel("Precision")
        plt.title("Training and Validation Precision")
        plt.legend()
    
    if recall is not None and val_recall is not None:
        plt.subplot(5, 1, 5)
        plt.plot(history.epoch, recall, label="Training Recall", linestyle='--')
        plt.plot(history.epoch, val_recall, label="Validation Recall", linestyle='--')
        plt.xlabel("Epochs")
        plt.ylabel("Recall")
        plt.title("Training and Validation Recall")
        plt.legend()
    
    plt.tight_layout()
    plt.show()
