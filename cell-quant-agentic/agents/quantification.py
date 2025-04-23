import numpy as np
from typing import List, Dict, Tuple, Any, Optional
import pandas as pd
from sklearn.cluster import DBSCAN
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
from PIL import Image
import torch

class QuantificationAgent:
    """
    Agent responsible for analyzing and quantifying cell detection results.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the QuantificationAgent."""
        self.config = config or {}
        
        # Initialize Phi-2
        self.phi2_tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
        self.phi2_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2")
        
        # Initialize LLaVA
        model_path = "liuhaotian/llava-v1.5-13b"
        model_name = get_model_name_from_path(model_path)
        self.llava_tokenizer, self.llava_model, self.llava_image_processor, self.llava_context_len = load_pretrained_model(
            model_path, None, model_name
        )
        
        # Initialize Mistral
        self.mistral_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
        self.mistral_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
        
    def analyze_detections(self, detections: List[List[Dict]], 
                         tile_coords: List[Tuple[int, int]] = None) -> Dict:
        """
        Analyze detection results across all tiles.
        
        Args:
            detections: List of detection results for each tile
            tile_coords: Optional list of tile coordinates in the original image
            
        Returns:
            Dict: Analysis results including counts and spatial statistics
        """
        # Count total detections
        total_count = sum(len(tile_dets) for tile_dets in detections)
        
        # Calculate confidence statistics
        all_confidences = []
        for tile_dets in detections:
            all_confidences.extend([d['confidence'] for d in tile_dets])
            
        confidence_stats = {
            'mean': np.mean(all_confidences) if all_confidences else 0,
            'std': np.std(all_confidences) if all_confidences else 0,
            'min': min(all_confidences) if all_confidences else 0,
            'max': max(all_confidences) if all_confidences else 0
        }
        
        # Analyze spatial distribution if tile coordinates provided
        spatial_stats = {}
        if tile_coords:
            # Combine detections with global coordinates
            global_points = []
            for (tile_x, tile_y), tile_dets in zip(tile_coords, detections):
                for det in tile_dets:
                    center = det['center']
                    global_x = tile_x + center[0]
                    global_y = tile_y + center[1]
                    global_points.append([global_x, global_y])
                    
            if global_points:
                # Perform clustering
                clustering = DBSCAN(eps=100, min_samples=3).fit(global_points)
                n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
                
                spatial_stats = {
                    'n_clusters': n_clusters,
                    'points_in_clusters': np.sum(clustering.labels_ != -1),
                    'isolated_points': np.sum(clustering.labels_ == -1)
                }
        
        return {
            'total_count': total_count,
            'confidence_stats': confidence_stats,
            'spatial_stats': spatial_stats,
            'per_tile_counts': [len(tile_dets) for tile_dets in detections]
        }
        
    def generate_report(self, analysis_results: Dict, output_dir: str = "results"):
        """
        Generate analysis report in multiple formats.
        
        Args:
            analysis_results: Results from analyze_detections
            output_dir: Directory to save report files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save JSON report
        json_path = os.path.join(output_dir, "analysis_report.json")
        with open(json_path, 'w') as f:
            json.dump(analysis_results, f, indent=2)
            
        # Create DataFrame for tile counts
        df = pd.DataFrame({
            'tile_index': range(len(analysis_results['per_tile_counts'])),
            'cell_count': analysis_results['per_tile_counts']
        })
        
        # Save CSV report
        csv_path = os.path.join(output_dir, "tile_counts.csv")
        df.to_csv(csv_path, index=False)
        
        # Generate summary text
        summary = [
            "Cell Detection Analysis Summary",
            "============================",
            f"Total cells detected: {analysis_results['total_count']}",
            "",
            "Confidence Statistics:",
            f"- Mean: {analysis_results['confidence_stats']['mean']:.3f}",
            f"- Std: {analysis_results['confidence_stats']['std']:.3f}",
            f"- Range: [{analysis_results['confidence_stats']['min']:.3f}, {analysis_results['confidence_stats']['max']:.3f}]",
            ""
        ]
        
        if analysis_results['spatial_stats']:
            summary.extend([
                "Spatial Analysis:",
                f"- Number of clusters: {analysis_results['spatial_stats']['n_clusters']}",
                f"- Cells in clusters: {analysis_results['spatial_stats']['points_in_clusters']}",
                f"- Isolated cells: {analysis_results['spatial_stats']['isolated_points']}"
            ])
            
        # Save summary text
        txt_path = os.path.join(output_dir, "summary.txt")
        with open(txt_path, 'w') as f:
            f.write('\n'.join(summary))
            
    def calculate_density_map(self, detections: List[List[Dict]], 
                            tile_coords: List[Tuple[int, int]],
                            image_size: Tuple[int, int],
                            sigma: float = 50.0) -> np.ndarray:
        """
        Calculate cell density heatmap.
        
        Args:
            detections: List of detection results for each tile
            tile_coords: List of tile coordinates
            image_size: Size of the original image
            sigma: Gaussian kernel standard deviation
            
        Returns:
            np.ndarray: Density heatmap
        """
        # Create empty density map
        density_map = np.zeros(image_size)
        
        # Add Gaussian for each detection
        for (tile_x, tile_y), tile_dets in zip(tile_coords, detections):
            for det in tile_dets:
                center = det['center']
                x = int(tile_x + center[0])
                y = int(tile_y + center[1])
                
                # Skip if outside image bounds
                if x < 0 or x >= image_size[1] or y < 0 or y >= image_size[0]:
                    continue
                    
                # Create Gaussian kernel
                x_grid, y_grid = np.meshgrid(
                    np.arange(max(0, x-int(3*sigma)), min(image_size[1], x+int(3*sigma))),
                    np.arange(max(0, y-int(3*sigma)), min(image_size[0], y+int(3*sigma)))
                )
                gaussian = np.exp(-((x_grid-x)**2 + (y_grid-y)**2) / (2*sigma**2))
                
                # Add to density map
                density_map[
                    max(0, y-int(3*sigma)):min(image_size[0], y+int(3*sigma)),
                    max(0, x-int(3*sigma)):min(image_size[1], x+int(3*sigma))
                ] += gaussian
                
        return density_map 

    def analyze_cell_distribution(self, detections: List[Dict[str, Any]], 
                                image_size: Tuple[int, int]) -> Dict[str, Any]:
        """Analyze the spatial distribution of cells."""
        # Extract cell centers
        centers = np.array([[d['center'][0], d['center'][1]] for d in detections])
        
        # Calculate density using DBSCAN
        dbscan = DBSCAN(eps=50, min_samples=5)
        clusters = dbscan.fit_predict(centers)
        
        # Calculate statistics
        n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
        n_noise = list(clusters).count(-1)
        
        return {
            'total_cells': len(detections),
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'clusters': clusters.tolist()
        }
        
    def explain_distribution(self, analysis: Dict[str, Any], image: np.ndarray) -> str:
        """Generate a natural language explanation of the cell distribution."""
        # Convert image to PIL format
        image_pil = Image.fromarray(image)
        
        # Prepare prompt for LLaVA
        prompt = f"""
        Analyze this microscopy image and explain the cell distribution:
        - Total cells: {analysis['total_cells']}
        - Number of clusters: {analysis['n_clusters']}
        - Number of isolated cells: {analysis['n_noise']}
        
        Please provide a detailed explanation of the cell distribution pattern and any notable observations.
        """
        
        # Get LLaVA analysis
        llava_response = eval_model(
            self.llava_model,
            self.llava_tokenizer,
            self.llava_image_processor,
            prompt,
            image_pil,
            self.llava_context_len
        )
        
        return llava_response
        
    def detect_trends(self, data: pd.DataFrame) -> str:
        """Detect and explain trends in the data using Phi-2."""
        # Prepare data summary
        data_summary = data.describe().to_string()
        
        # Prepare prompt for Phi-2
        prompt = f"""
        Analyze the following cell quantification data and identify any significant trends or patterns:
        
        {data_summary}
        
        Please provide a detailed analysis of the trends, including:
        1. Overall patterns in cell counts
        2. Any correlations between different measurements
        3. Notable outliers or anomalies
        4. Potential biological implications
        """
        
        # Generate analysis with Phi-2
        inputs = self.phi2_tokenizer(prompt, return_tensors="pt")
        outputs = self.phi2_model.generate(**inputs, max_length=500)
        analysis = self.phi2_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return analysis
        
    def generate_insights(self, image: np.ndarray, detections: List[Dict[str, Any]], 
                         analysis: Dict[str, Any]) -> str:
        """Generate comprehensive insights using Mistral."""
        # Convert image to PIL format
        image_pil = Image.fromarray(image)
        
        # Prepare prompt for Mistral
        prompt = f"""
        Based on the following cell analysis results, provide comprehensive insights:
        
        Image Analysis:
        - Total cells detected: {analysis['total_cells']}
        - Cell distribution: {analysis['n_clusters']} clusters with {analysis['n_noise']} isolated cells
        
        Detection Details:
        {json.dumps(detections[:5], indent=2)}  # First 5 detections as example
        
        Please provide:
        1. A detailed interpretation of the results
        2. Potential biological significance
        3. Any concerns or notable observations
        4. Recommendations for further analysis
        """
        
        # Generate insights with Mistral
        inputs = self.mistral_tokenizer(prompt, return_tensors="pt")
        outputs = self.mistral_model.generate(**inputs, max_length=1000)
        insights = self.mistral_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return insights
        
    def process_results(self, image: np.ndarray, detections: List[Dict[str, Any]], 
                       image_size: Tuple[int, int]) -> Dict[str, Any]:
        """Process detection results and generate comprehensive analysis."""
        # Analyze cell distribution
        distribution = self.analyze_cell_distribution(detections, image_size)
        
        # Generate explanations
        distribution_explanation = self.explain_distribution(distribution, image)
        
        # Create DataFrame for trend analysis
        data = pd.DataFrame({
            'x': [d['center'][0] for d in detections],
            'y': [d['center'][1] for d in detections],
            'confidence': [d['confidence'] for d in detections]
        })
        
        # Detect trends
        trends = self.detect_trends(data)
        
        # Generate insights
        insights = self.generate_insights(image, detections, distribution)
        
        return {
            'distribution': distribution,
            'distribution_explanation': distribution_explanation,
            'trends': trends,
            'insights': insights
        } 