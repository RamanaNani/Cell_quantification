import openslide
from PIL import Image
import numpy as np
from typing import Dict, Tuple, List, Any, Optional
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from lxml import etree
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION

class FileHandlerAgent:
    """
    Agent responsible for handling file operations (VSI and XML files).
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the FileHandlerAgent.
        
        Args:
            config (Optional[Dict[str, Any]], optional): Configuration for the agent
        """
        self.config = config or {}
        self.vsi_file = None
        self.xml_file = None
        self.metadata = {}
        self.markers = []
        
        # Initialize BLIP-2
        self.blip2_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.blip2_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
        
        # Initialize MiniGPT-4
        cfg = Config()
        cfg.model_cfg = "minigpt4/configs/minigpt4.yaml"
        cfg.run_cfg = "minigpt4/configs/run.yaml"
        self.minigpt4 = Chat(cfg)
        
    def load_vsi(self, vsi_path: str) -> Dict[str, Any]:
        """Load a VSI file and extract metadata."""
        try:
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
        except Exception as e:
            raise Exception(f"Failed to load VSI file: {str(e)}")
            
    def load_xml(self, xml_path: str) -> List[Dict[str, Any]]:
        """Load an XML file containing marker information."""
        try:
            tree = etree.parse(xml_path)
            root = tree.getroot()
            
            # Extract markers
            self.markers = []
            for marker in root.findall('.//Marker'):
                marker_data = {
                    'id': marker.get('Id'),
                    'x': float(marker.get('X')),
                    'y': float(marker.get('Y')),
                    'type': marker.get('Type'),
                    'description': marker.get('Description', '')
                }
                self.markers.append(marker_data)
                
            return self.markers
        except Exception as e:
            raise Exception(f"Failed to load XML file: {str(e)}")
            
    def read_region(self, x: int, y: int, width: int, height: int, level: int = 0) -> np.ndarray:
        """Read a region from the VSI file."""
        try:
            region = self.vsi_file.read_region((x, y), level, (width, height))
            return np.array(region)
        except Exception as e:
            raise Exception(f"Failed to read region: {str(e)}")
            
    def get_caption(self, image: np.ndarray, model: str = 'blip2') -> str:
        """Generate a caption for an image using either BLIP-2 or MiniGPT-4."""
        if model == 'blip2':
            # Convert image to PIL format
            image_pil = Image.fromarray(image)
            
            # Generate caption with BLIP-2
            inputs = self.blip2_processor(image_pil, return_tensors="pt")
            out = self.blip2_model.generate(**inputs)
            caption = self.blip2_processor.decode(out[0], skip_special_tokens=True)
            
        else:  # MiniGPT-4
            # Convert image to PIL format
            image_pil = Image.fromarray(image)
            
            # Generate caption with MiniGPT-4
            self.minigpt4.upload_img(image_pil)
            self.minigpt4.ask("Describe this image in detail.")
            caption = self.minigpt4.answer()
            
        return caption
        
    def get_tile_captions(self, tile_size: int = 512, overlap: int = 0) -> List[Dict[str, Any]]:
        """Generate captions for all tiles in the image."""
        if not self.vsi_file:
            raise Exception("No VSI file loaded")
            
        captions = []
        width, height = self.vsi_file.dimensions
        
        for y in range(0, height, tile_size - overlap):
            for x in range(0, width, tile_size - overlap):
                try:
                    # Read tile
                    tile = self.read_region(x, y, tile_size, tile_size)
                    
                    # Generate caption
                    caption = self.get_caption(tile)
                    
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
        """Close the VSI file."""
        if self.vsi_file:
            self.vsi_file.close()
            self.vsi_file = None
            
    def __enter__(self):
        """Context manager entry"""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close() 