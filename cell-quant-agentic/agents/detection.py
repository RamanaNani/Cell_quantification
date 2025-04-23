import numpy as np
import cv2
from typing import List, Tuple, Dict, Any, Optional
from PIL import Image
import torch
import torchvision
from pathlib import Path
import os
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
from medsam import MedSAM
from seggpt import SegGPT

class DetectionAgent:
    """
    Agent responsible for detecting cells in preprocessed images.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize YOLOv8
        self.yolo_model = YOLO('yolov8n.pt')
        
        # Initialize SAM
        sam_checkpoint = "sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        self.sam_predictor = SamPredictor(sam)
        
        # Initialize MedSAM
        self.medsam = MedSAM()
        
        # Initialize SegGPT
        self.seggpt = SegGPT()
        
    def detect_cells_yolo(self, image: np.ndarray, confidence: float = 0.5) -> List[Dict[str, Any]]:
        """Detect cells using YOLOv8."""
        # Convert to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
        # Run YOLO detection
        results = self.yolo_model(image, conf=confidence)
        
        # Process results
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = box.cls[0].cpu().numpy()
                
                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': float(conf),
                    'class': int(cls),
                    'class_name': self.yolo_model.names[int(cls)]
                })
                
        return detections
        
    def segment_cells_sam(self, image: np.ndarray, points: Optional[List[Tuple[int, int]]] = None) -> np.ndarray:
        """Segment cells using SAM."""
        # Convert to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
        # Set image
        self.sam_predictor.set_image(image)
        
        if points:
            # Convert points to numpy array
            input_points = np.array(points)
            input_labels = np.ones(len(points))
            
            # Predict masks
            masks, scores, _ = self.sam_predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=True
            )
            
            # Use the best mask
            best_mask = masks[np.argmax(scores)]
        else:
            # Generate automatic masks
            masks = self.sam_predictor.generate(image)
            best_mask = masks[0]  # Use the first mask
            
        return best_mask.astype(np.uint8) * 255
        
    def segment_cells_medsam(self, image: np.ndarray) -> np.ndarray:
        """Segment cells using MedSAM."""
        # Convert to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
        # Run MedSAM segmentation
        mask = self.medsam.segment(image)
        return mask
        
    def segment_cells_seggpt(self, image: np.ndarray, reference_image: Optional[np.ndarray] = None) -> np.ndarray:
        """Segment cells using SegGPT."""
        # Convert to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
        if reference_image is not None and len(reference_image.shape) == 2:
            reference_image = cv2.cvtColor(reference_image, cv2.COLOR_GRAY2RGB)
            
        # Run SegGPT segmentation
        mask = self.seggpt.segment(image, reference_image)
        return mask
        
    def detect_and_segment(self, image: np.ndarray, method: str = 'yolo', 
                         confidence: float = 0.5) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """Detect and segment cells using the specified method."""
        # Detect cells
        if method == 'yolo':
            detections = self.detect_cells_yolo(image, confidence)
            points = [(int((box['bbox'][0] + box['bbox'][2])/2), 
                      int((box['bbox'][1] + box['bbox'][3])/2)) 
                     for box in detections]
            mask = self.segment_cells_sam(image, points)
        elif method == 'sam':
            detections = []
            mask = self.segment_cells_sam(image)
        elif method == 'medsam':
            detections = []
            mask = self.segment_cells_medsam(image)
        elif method == 'seggpt':
            detections = []
            mask = self.segment_cells_seggpt(image)
        else:
            raise ValueError(f"Unknown method: {method}")
            
        return detections, mask
        
    def process_image(self, image: np.ndarray, method: str = 'yolo', 
                     confidence: float = 0.5) -> Dict[str, Any]:
        """Process an image to detect and segment cells."""
        # Detect and segment cells
        detections, mask = self.detect_and_segment(image, method, confidence)
        
        # Count cells
        cell_count = len(detections) if detections else np.sum(mask > 0) // 1000  # Approximate count from mask
        
        return {
            'detections': detections,
            'mask': mask,
            'cell_count': cell_count
        }
        
    def process_batch(self, images: List[np.ndarray], output_dir: str = None) -> List[List[Dict]]:
        """
        Process a batch of images.
        
        Args:
            images: List of preprocessed images
            output_dir: Optional directory to save visualization results
            
        Returns:
            List[List[Dict]]: Detection results for each image
        """
        batch_results = []
        
        for i, image in enumerate(images):
            try:
                # Detect cells
                detections = self.detect_cells_yolo(image)
                batch_results.append(detections)
                
                # Visualize if output directory provided
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    vis_image = self.visualize_detections(image, detections)
                    output_path = os.path.join(output_dir, f"detection_{i:04d}.png")
                    Image.fromarray((vis_image * 255).astype(np.uint8)).save(output_path)
                    
            except Exception as e:
                print(f"Warning: Failed to process image {i}: {str(e)}")
                batch_results.append([])
                
        return batch_results
        
    def visualize_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Visualize detection results on the image.
        
        Args:
            image: Input image
            detections: List of detection results
            
        Returns:
            np.ndarray: Image with visualized detections
        """
        # Make a copy for visualization
        vis_image = image.copy()
        
        # Draw each detection
        for det in detections:
            box = det['bbox']
            conf = det['confidence']
            
            # Draw bounding box
            cv2.rectangle(
                vis_image,
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                (0, 1, 0),  # Green
                2
            )
            
            # Draw confidence score
            cv2.putText(
                vis_image,
                f"{conf:.2f}",
                (int(box[0]), int(box[1] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 1, 0),
                1
            )
            
        return vis_image 