import os
import javabridge  # For initializing the JVM for bioformats
import bioformats  # For reading VSI files
import numpy as np
from typing import Dict, Any  # For type hinting
import matplotlib.pyplot as plt  # For displaying the image

# Start the JVM for bioformats
javabridge.start_vm(class_path=bioformats.JARS)

class VSIFileHandler:
    """
    Class responsible for handling VSI file operations.
    """

    def __init__(self):
        self.vsi_file = None
        self.metadata = {}

    def load_vsi(self, vsi_path: str) -> Dict[str, Any]:
        """Load VSI file using bioformats."""
        if not os.path.exists(vsi_path):
            raise FileNotFoundError(f"File does not exist: {vsi_path}")

        try:
            # Read the VSI file using bioformats
            reader = bioformats.ImageReader(vsi_path)
            image = reader.read()  # Load the image as a numpy array

            # Extract metadata from VSI file
            self.metadata = {
                'width': image.shape[1],  # Get width from image shape
                'height': image.shape[0],  # Get height from image shape
                'level_count': 1,  # Assuming a single level for simplicity
                'level_dimensions': [image.shape[1], image.shape[0]],  # Image dimensions
                'level_downsamples': [1],  # No downsample
                'properties': {}  # Add any additional metadata properties if needed
            }
            self.vsi_file = image
            return self.metadata
        except Exception as e:
            raise Exception(f"Failed to load VSI image using bioformats: {str(e)}")

    def get_image(self) -> np.ndarray:
        """Return the loaded VSI image."""
        if self.vsi_file is None:
            raise Exception("No VSI file loaded")
        return self.vsi_file

    def close(self):
        """Cleanup resources."""
        self.vsi_file = None
