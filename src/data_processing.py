import os
import cv2
import numpy as np
from typing import List, Tuple
from PIL import Image


def load_images_masks (image_dir: str, mask_dir: str, img_size: Tuple[int, int] = (128, 128)) -> Tuple[
    np.array, np.array]:
    """
    Loads images and corresponding masks from specified directories, resizes them, and normalizes pixel values.

    Parameters:
    ----------
    image_dir : str
        Directory containing the images.
    mask_dir : str
        Directory containing the masks with the same filenames as the images.
    img_size : Tuple[int, int], optional
        Desired size for resizing the images and masks (default is (128, 128)).

    Returns:
    -------
    Tuple[np.array, np.array]
        A tuple containing two numpy arrays:
        - Array of images, normalized to [0, 1] and resized to img_size.
        - Array of corresponding masks, normalized to [0, 1], resized to img_size, and reshaped to have one channel.
    """
    image_list: List = []
    mask_list: List = []
    
    for image_name in os.listdir(image_dir):
        img_path = os.path.join(image_dir, image_name)
        mask_path = os.path.join(mask_dir, image_name)
        
        img = cv2.imread(img_path)
        img = cv2.resize(img, img_size)
        img = img / 255.0
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, img_size)
        mask = mask / 255.0
        
        image_list.append(img)
        mask_list.append(mask)
    
    images = np.array(image_list)
    masks = np.array(mask_list).reshape(-1, img_size[0], img_size[1], 1)
    
    return images, masks


def load_single_image (image_path: str, greyscale: bool = False) -> np.array:
    """
    Loads a single image from a specified path and converts it to grayscale if needed.

    Parameters:
    ----------
    image_path : str
        Path to the image file.
    greyscale : bool, optional
        Whether to convert the image to grayscale (default is False).

    Returns:
    -------
    np.array
        The loaded image as a numpy array. If greyscale is True, the image is converted to grayscale.
    """
    img = Image.open(image_path)
    if greyscale:
        img = img.convert('L')
    return np.array(img)


def load_single_mask (mask_path: str) -> np.array:
    """
    Loads a single mask from a specified path and converts it to grayscale.

    Parameters:
    ----------
    mask_path : str
        Path to the mask file.

    Returns:
    -------
    np.array
        The loaded mask as a grayscale numpy array.
    """
    mask = Image.open(mask_path).convert('L')
    return np.array(mask)
