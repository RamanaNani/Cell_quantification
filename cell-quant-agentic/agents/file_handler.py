import sys
import os
import openslide # type: ignore
from dotenv import load_dotenv
from PIL import Image
import numpy as np
from typing import Dict, Tuple, List, Any, Optional
import xml.etree.ElementTree as ET
from pathlib import Path
from lxml import etree
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
import openai
import base64
from io import BytesIO
from PIL import Image as PILImage

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_openai_caption(image: Image.Image) -> str:
    """Generate a caption using OpenAI's GPT-4o model."""

    client = openai.OpenAI()

    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    response = client.chat.completions.create(
        model="gpt-4o",   # <-- use gpt-4o now
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this pathology image in detail."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_str}"}}
                ]
            }
        ],
        max_tokens=200
    )

    return response.choices[0].message.content

class FileHandlerAgent:
    """
    Agent responsible for handling file operations (VSI and XML files).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.vsi_file = None
        self.xml_file = None
        self.metadata = {}
        self.markers = []

        # Initialize BLIP-2
        self.blip2_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.blip2_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")


    def load_vsi(self, vsi_path: str) -> Dict[str, Any]:
        if not os.path.exists(vsi_path):
            raise FileNotFoundError(f"File does not exist: {vsi_path}")

        try:
            # First try to load with OpenSlide
            self.vsi_file = openslide.OpenSlide(vsi_path)
            self.metadata = {
                'width': self.vsi_file.dimensions[0],
                'height': self.vsi_file.dimensions[1],
                'level_count': self.vsi_file.level_count,
                'level_dimensions': self.vsi_file.level_dimensions,
                'level_downsamples': self.vsi_file.level_downsamples,
                'properties': dict(self.vsi_file.properties)
            }
            return self.metadata

        except openslide.OpenSlideUnsupportedFormatError:
            try:
                # Fallback to PIL for unsupported formats like PNG, JPG, TIFF
                img = PILImage.open(vsi_path).convert("RGB")
                self.vsi_file = img
                self.metadata = {
                    'width': img.width,
                    'height': img.height,
                    'level_count': 1,
                    'level_dimensions': [img.size],
                    'level_downsamples': [1],
                    'properties': {}
                }
                return self.metadata
            except Exception as fallback_error:
                raise Exception(f"File is not supported by OpenSlide or PIL: {fallback_error}")

        except Exception as e:
            raise Exception(f"Failed to load image file: {str(e)}")



    def load_xml(self, xml_path: str) -> List[Dict[str, Any]]:
        """Load an XML file containing marker information."""
        try:
            tree = etree.parse(xml_path)
            root = tree.getroot()
            self.markers = []

            for marker in root.findall('.//Marker'):
                x = marker.get('X')
                y = marker.get('Y')
                if x is None or y is None:
                    continue  # Skip markers without valid coordinates

                marker_data = {
                    'id': marker.get('Id'),
                    'x': float(x),
                    'y': float(y),
                    'type': marker.get('Type'),
                    'description': marker.get('Description', '')
                }
                self.markers.append(marker_data)

            return self.markers
        except Exception as e:
            raise Exception(f"Failed to load XML file: {str(e)}")


    def read_region(self, x: int, y: int, width: int, height: int, level: int = 0) -> np.ndarray:
        """Read a region from the VSI file or fallback PIL image."""
        try:
            if hasattr(self.vsi_file, "read_region"):  # OpenSlide object
                region = self.vsi_file.read_region((x, y), level, (width, height)).convert("RGB")
            else:  # PIL Image fallback
                region = self.vsi_file.crop((x, y, x + width, y + height))
            return np.array(region)
        except Exception as e:
            raise Exception(f"Failed to read region: {str(e)}")



    def get_caption(self, image: np.ndarray, model: str = 'blip2') -> str:
        image_pil = Image.fromarray(image)
        if model == 'blip2':
            inputs = self.blip2_processor(image_pil, return_tensors="pt")
            out = self.blip2_model.generate(**inputs)
            return self.blip2_processor.decode(out[0], skip_special_tokens=True)
        elif model == 'openai':
            return generate_openai_caption(image_pil)
        else:
            raise ValueError(f"Unsupported model: {model}")

    def get_tile_captions(self, tile_size: int = 512, overlap: int = 0) -> List[Dict[str, Any]]:
        """Generate captions for all tiles in the image."""
        if not self.vsi_file:
            raise Exception("No VSI file loaded")

        captions = []

        if hasattr(self.vsi_file, "dimensions"):  # OpenSlide object
            width, height = self.vsi_file.dimensions
        else:  # PIL fallback
            width, height = self.vsi_file.size

        for y in range(0, height, tile_size - overlap):
            for x in range(0, width, tile_size - overlap):
                try:
                    # Read tile
                    tile = self.read_region(x, y, tile_size, tile_size)

                    # Generate caption
                    caption = self.get_caption(tile, model=self.config.get("caption_model", "blip2"))

                    captions.append({
                        'x': x,
                        'y': y,
                        'width': tile_size,
                        'height': tile_size,
                        'caption': caption
                    })
                except Exception as e:
                    print(f"Failed to process tile at ({x}, {y}): {str(e)}")

        return captions


    def close(self):
        if self.vsi_file:
            self.vsi_file.close()
            self.vsi_file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

