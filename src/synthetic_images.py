from typing import Tuple, List
import numpy as np
import matplotlib.pyplot as plt


class ImageGenerator:
    """
    A class to generate synthetic binary images with crack patterns.

    Attributes:
    ----------
    image_size : Tuple[int, int]
        The size of the generated images.
    circle_center : List[Tuple]
        List of centers of circular cracks generated.
    circle_radius : List[int]
        List of radii of circular cracks generated.
    amplitudes : List[Tuple]
        List of amplitudes used in sine and cosine cracks.
    frequencies : List[Tuple]
        List of frequencies used in sine and cosine cracks.
    phases : List[Tuple]
        List of phase shifts used in sine and cosine cracks.
    """
    
    def __init__ (self, image_size: Tuple[int, int]):
        """
        Initializes the ImageGenerator with a specified image size.

        Parameters:
        ----------
        image_size : Tuple[int, int]
            Size of the image to generate.
        """
        self.image_size = image_size
        self.circle_center: List[Tuple] = []
        self.circle_radius: List[int] = []
        self.amplitudes: List[Tuple] = []
        self.frequencies: List[Tuple] = []
        self.phases: List[Tuple] = []
    
    def create_circular_crack (self, center: Tuple[float, float], radius: int, num_cracks: int = 10) -> np.array:
        """
        Creates a binary image with circular cracks radiating from a central point.

        Parameters:
        ----------
        center : Tuple[float, float]
            The center of the circular cracks.
        radius : int
            The radius of the cracks.
        num_cracks : int, optional
            The number of cracks radiating from the center (default is 10).

        Returns:
        -------
        np.array
            Binary image with circular cracks.
        """
        image = np.zeros(self.image_size, dtype=np.uint8)
        angles = np.linspace(0, 2 * np.pi, num_cracks, endpoint=False)
        self.circle_center.append(center)
        self.circle_radius.append(radius)
        for angle in angles:
            for r in range(radius):
                x = int(center[0] + r * np.cos(angle))
                y = int(center[1] + r * np.sin(angle))
                
                if np.random.rand() > 0.2:
                    thickness = np.random.randint(1, 3)
                    for t in range(-thickness, thickness + 1):
                        if 0 <= x + t < self.image_size[1] and 0 <= y + t < self.image_size[0]:
                            image[y + t, x] = 1
        
        return image
    
    def create_sine_cosine_crack (self) -> np.array:
        """
        Creates a binary image with cracks using sine and cosine waves.

        Returns:
        -------
        np.array
            Binary image with sine and cosine cracks.
        """
        image: np.array = np.zeros(self.image_size)
        x: np.array = np.linspace(0, self.image_size[1] - 1, self.image_size[1])
        
        frequency_sin: float = np.random.uniform(0.01, 0.1)
        frequency_cos: float = np.random.uniform(0.01, 0.1)
        amplitude_sin: float = np.random.uniform(5, 20)
        amplitude_cos: float = np.random.uniform(5, 20)
        phase_shift_sin: float = np.random.rand() * 2 * np.pi
        phase_shift_cos: float = np.random.rand() * 2 * np.pi
        
        self.amplitudes.append((amplitude_sin, amplitude_cos))
        self.frequencies.append((frequency_sin, frequency_cos))
        self.phases.append((phase_shift_sin, phase_shift_cos))
        
        noise_amplitude = np.random.uniform(1, 3)
        noise = noise_amplitude * np.random.normal(size=self.image_size[1])
        
        y = (amplitude_sin * np.sin(frequency_sin * x + phase_shift_sin) +
             amplitude_cos * np.cos(frequency_cos * x + phase_shift_cos) +
             (self.image_size[0] // 2) + noise)
        
        y = np.clip(y, 0, self.image_size[0] - 1).astype(int)
        crack_thickness: int = np.random.randint(1, 4)
        for i in range(self.image_size[1]):
            for j in range(-crack_thickness, crack_thickness + 1):
                if 0 <= y[i] + j < self.image_size[0]:
                    image[y[i] + j, i] = 1
        
        return image
    
    def create_synthetic_images (self, num_images: int = 10) -> List[np.ndarray]:
        """
        Generates a list of synthetic binary images with either circular or sine-cosine cracks.

        Parameters:
        ----------
        num_images : int, optional
            Number of images to generate (default is 10).

        Returns:
        -------
        List[np.ndarray]
            List of generated binary images with cracks.
        """
        images: List[np.array] = []
        for _ in range(num_images):
            if np.random.rand() > 0.5:
                center = (
                np.random.randint(20, self.image_size[1] - 20), np.random.randint(20, self.image_size[0] - 20))
                radius = np.random.randint(5, 20)
                image = self.create_circular_crack(center, radius)
            else:
                image = self.create_sine_cosine_crack()
            
            images.append(image)
        
        return images
    
    def create_composite_synthetic_images (self, num_images: int = 10, composition_limit: int = 3) -> List[np.array]:
        """
        Generates composite images by superimposing multiple crack patterns.

        Parameters:
        ----------
        num_images : int, optional
            Number of composite images to generate (default is 10).
        composition_limit : int, optional
            Maximum number of cracks per composite image (default is 3).

        Returns:
        -------
        List[np.array]
            List of generated composite binary images with cracks.
        """
        images: List[np.array] = []
        for _ in range(num_images):
            composite_image = np.zeros(self.image_size, dtype=np.uint8)
            initial_images: List[np.array] = self.create_synthetic_images(composition_limit)
            
            for img in initial_images:
                composite_image = np.logical_or(composite_image, img).astype(np.uint8)
            
            images.append(composite_image)
        return images
    
    @staticmethod
    def display_images (images: List[np.array]) -> None:
        """
        Displays a grid of the generated images using matplotlib.

        Parameters:
        ----------
        images : List[np.array]
            List of images to display.
        """
        plt.figure(figsize=(15, 10))
        for i, img in enumerate(images):
            plt.subplot(3, 4, i + 1)
            plt.imshow(img, cmap="gray", vmin=0, vmax=1)
            plt.axis("off")
            plt.title(f'Image {i + 1}')
        plt.tight_layout()
        plt.show()
    
    def get_circle_center (self) -> List[Tuple]:
        """
        Returns the centers of circular cracks generated.

        Returns:
        -------
        List[Tuple]
            List of circle centers.
        """
        return self.circle_center
    
    def get_circle_radius (self) -> List[int]:
        """
        Returns the radii of circular cracks generated.

        Returns:
        -------
        List[int]
            List of circle radii.
        """
        return self.circle_radius
    
    def get_frequencies (self) -> List[Tuple]:
        """
        Returns the frequencies used in sine and cosine cracks.

        Returns:
        -------
        List[Tuple]
            List of sine and cosine frequencies.
        """
        return self.frequencies
    
    def get_amplitudes (self) -> List[Tuple]:
        """
        Returns the amplitudes used in sine and cosine cracks.

        Returns:
        -------
        List[Tuple]
            List of sine and cosine amplitudes.
        """
        return self.amplitudes
    
    def get_phases (self) -> List[Tuple]:
        """
        Returns the phase shifts used in sine and cosine cracks.

        Returns:
        -------
        List[Tuple]
            List of sine and cosine phase shifts.
        """
        return self.phases
