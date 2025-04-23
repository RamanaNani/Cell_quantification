import numpy as np
from PIL import Image
import cv2
from typing import List, Tuple, Union, Dict, Any, Optional
import os
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
import torch
from transformers import CLIPProcessor, CLIPModel

class PreprocessingAgent:
    """
    Agent responsible for preprocessing image tiles before detection.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the PreprocessingAgent."""
        self.config = config or {}
        
        # Initialize BLIP-2
        self.blip2_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.blip2_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
        
        # Initialize LLaVA
        model_path = "liuhaotian/llava-v1.5-13b"
        model_name = get_model_name_from_path(model_path)
        self.llava_tokenizer, self.llava_model, self.llava_image_processor, self.llava_context_len = load_pretrained_model(
            model_path, None, model_name
        )
        
        # Initialize CLIP
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        
    def normalize_image(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """
        Normalize image to prepare for cell detection.
        
        Args:
            image: Input image (numpy array or PIL Image)
            
        Returns:
            np.ndarray: Normalized image
        """
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
            
        # Convert to float32
        image = image.astype(np.float32)
        
        # Normalize to [0, 1]
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        
        return image
        
    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image contrast using CLAHE.
        
        Args:
            image: Input image
            
        Returns:
            np.ndarray: Contrast-enhanced image
        """
        if len(image.shape) == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            
            # Merge channels and convert back to RGB
            enhanced_lab = cv2.merge((cl, a, b))
            enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        else:
            # Apply CLAHE to grayscale image
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(image)
            
        return enhanced
        
    def preprocess_tile(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """
        Apply full preprocessing pipeline to an image tile.
        
        Args:
            image: Input image tile
            
        Returns:
            np.ndarray: Preprocessed image
        """
        # Convert to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
            
        # Ensure RGB format
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
        # Normalize
        image = self.normalize_image(image)
        
        # Enhance contrast
        image = self.enhance_contrast(image)
        
        return image
        
    def preprocess_batch(self, image_paths: List[str], output_dir: str = None) -> List[np.ndarray]:
        """
        Preprocess a batch of images.
        
        Args:
            image_paths: List of paths to image files
            output_dir: Optional directory to save preprocessed images
            
        Returns:
            List[np.ndarray]: List of preprocessed images
        """
        preprocessed = []
        
        for i, path in enumerate(image_paths):
            try:
                # Load image
                image = Image.open(path)
                
                # Preprocess
                processed = self.preprocess_tile(image)
                
                # Save if output directory provided
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    output_path = os.path.join(output_dir, f"preprocessed_{i:04d}.png")
                    Image.fromarray((processed * 255).astype(np.uint8)).save(output_path)
                
                preprocessed.append(processed)
                
            except Exception as e:
                print(f"Warning: Failed to preprocess {path}: {str(e)}")
                continue
                
        return preprocessed
        
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Apply basic preprocessing to an image."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
        # Normalize
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        image = clahe.apply(image)
        
        return image
        
    def is_blank(self, image: np.ndarray) -> bool:
        """Check if an image is blank using CLIP."""
        # Convert to PIL format
        image_pil = Image.fromarray(image)
        
        # Get CLIP embeddings
        inputs = self.clip_processor(images=image_pil, return_tensors="pt")
        image_features = self.clip_model.get_image_features(**inputs)
        
        # Compare with blank image features
        blank_image = np.zeros_like(image)
        blank_pil = Image.fromarray(blank_image)
        blank_inputs = self.clip_processor(images=blank_pil, return_tensors="pt")
        blank_features = self.clip_model.get_image_features(**blank_inputs)
        
        # Calculate similarity
        similarity = torch.cosine_similarity(image_features, blank_features)
        return similarity.item() > 0.95
        
    def is_blurry(self, image: np.ndarray, threshold: float = 100.0) -> bool:
        """Check if an image is blurry using Laplacian variance."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold
        
    def is_informative(self, image: np.ndarray) -> bool:
        """Check if an image contains useful information using BLIP-2 and LLaVA."""
        # Convert to PIL format
        image_pil = Image.fromarray(image)
        
        # Get BLIP-2 caption
        blip2_inputs = self.blip2_processor(image_pil, return_tensors="pt")
        blip2_out = self.blip2_model.generate(**blip2_inputs)
        blip2_caption = self.blip2_processor.decode(blip2_out[0], skip_special_tokens=True)
        
        # Get LLaVA analysis
        llava_prompt = "Does this image contain any cells or biological structures? Answer with yes or no."
        llava_response = eval_model(
            self.llava_model,
            self.llava_tokenizer,
            self.llava_image_processor,
            llava_prompt,
            image_pil,
            self.llava_context_len
        )
        
        # Check if either model indicates presence of cells
        return "cell" in blip2_caption.lower() or "yes" in llava_response.lower()
        
    def filter_regions(self, regions: List[np.ndarray]) -> List[np.ndarray]:
        """Filter out blank, blurry, or uninformative regions."""
        filtered_regions = []
        
        for region in regions:
            if (not self.is_blank(region) and 
                not self.is_blurry(region) and 
                self.is_informative(region)):
                filtered_regions.append(region)
                
        return filtered_regions
        
    def normalize_intensity(self, image: np.ndarray) -> np.ndarray:
        """Normalize image intensity."""
        if len(image.shape) == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Normalize L channel
            l_norm = cv2.normalize(l, None, 0, 255, cv2.NORM_MINMAX)
            
            # Merge channels and convert back to RGB
            norm_lab = cv2.merge((l_norm, a, b))
            normalized = cv2.cvtColor(norm_lab, cv2.COLOR_LAB2RGB)
        else:
            # Normalize grayscale image
            normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
            
        return normalized
        
    def process_regions(self, regions: List[np.ndarray]) -> List[np.ndarray]:
        """Process a list of image regions."""
        processed_regions = []
        
        for region in regions:
            # Apply preprocessing steps
            enhanced = self.enhance_contrast(region)
            normalized = self.normalize_intensity(enhanced)
            processed_regions.append(normalized)
            
        return processed_regions 