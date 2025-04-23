import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
import os
import json
import pandas as pd
from datetime import datetime
import seaborn as sns
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION
import torch

class ReportAgent:
    """
    Agent responsible for generating comprehensive analysis reports.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the ReportAgent."""
        self.style_setup()
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
        
        # Initialize Mistral
        self.mistral_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
        self.mistral_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
        
        # Initialize MiniGPT-4
        cfg = Config()
        cfg.model_cfg = "minigpt4/configs/minigpt4.yaml"
        cfg.run_cfg = "minigpt4/configs/run.yaml"
        self.minigpt4 = Chat(cfg)
        
    def style_setup(self):
        """Set up plotting style."""
        plt.style.use('seaborn')
        sns.set_palette("husl")
        
    def create_report(self, analysis_results: Dict, density_map: np.ndarray = None,
                     sample_images: List[Tuple[str, str]] = None, output_dir: str = "results"):
        """
        Create a comprehensive report with visualizations.
        
        Args:
            analysis_results: Results from QuantificationAgent
            density_map: Optional density heatmap
            sample_images: Optional list of (image_path, caption) pairs
            output_dir: Directory to save report files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Create report directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = os.path.join(output_dir, f"report_{timestamp}")
        os.makedirs(report_dir, exist_ok=True)
        
        # Generate plots
        self._plot_count_distribution(analysis_results, report_dir)
        self._plot_confidence_distribution(analysis_results, report_dir)
        if density_map is not None:
            self._plot_density_map(density_map, report_dir)
            
        # Generate HTML report
        html_content = self._generate_html_report(
            analysis_results,
            density_map is not None,
            sample_images,
            timestamp
        )
        
        with open(os.path.join(report_dir, "report.html"), 'w') as f:
            f.write(html_content)
            
        # Save raw data
        with open(os.path.join(report_dir, "analysis_results.json"), 'w') as f:
            json.dump(analysis_results, f, indent=2)
            
    def _plot_count_distribution(self, results: Dict, output_dir: str):
        """Plot distribution of cell counts per tile."""
        plt.figure(figsize=(10, 6))
        sns.histplot(results['per_tile_counts'], bins=30)
        plt.title("Distribution of Cell Counts per Tile")
        plt.xlabel("Number of Cells")
        plt.ylabel("Frequency")
        plt.savefig(os.path.join(output_dir, "count_distribution.png"))
        plt.close()
        
    def _plot_confidence_distribution(self, results: Dict, output_dir: str):
        """Plot distribution of detection confidences."""
        plt.figure(figsize=(10, 6))
        stats = results['confidence_stats']
        x = np.linspace(stats['min'], stats['max'], 100)
        plt.axvline(stats['mean'], color='r', linestyle='--', label=f"Mean ({stats['mean']:.3f})")
        plt.axvline(stats['mean'] - stats['std'], color='g', linestyle=':', label='±1 std')
        plt.axvline(stats['mean'] + stats['std'], color='g', linestyle=':')
        plt.title("Detection Confidence Statistics")
        plt.xlabel("Confidence Score")
        plt.ylabel("Density")
        plt.legend()
        plt.savefig(os.path.join(output_dir, "confidence_stats.png"))
        plt.close()
        
    def _plot_density_map(self, density_map: np.ndarray, output_dir: str):
        """Plot cell density heatmap."""
        plt.figure(figsize=(12, 8))
        plt.imshow(density_map, cmap='hot')
        plt.colorbar(label="Cell Density")
        plt.title("Cell Density Heatmap")
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, "density_map.png"))
        plt.close()
        
    def _generate_html_report(self, results: Dict, has_density_map: bool,
                            sample_images: List[Tuple[str, str]], timestamp: str) -> str:
        """Generate HTML report content."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Cell Detection Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .container {{ max-width: 1200px; margin: auto; }}
                .section {{ margin-bottom: 40px; }}
                .plot {{ margin: 20px 0; text-align: center; }}
                .plot img {{ max-width: 100%; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                .sample-images {{ display: flex; flex-wrap: wrap; gap: 20px; }}
                .sample-image {{ max-width: 300px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Cell Detection Analysis Report</h1>
                <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                
                <div class="section">
                    <h2>Summary Statistics</h2>
                    <table>
                        <tr><th>Metric</th><th>Value</th></tr>
                        <tr><td>Total Cells Detected</td><td>{results['total_count']}</td></tr>
                        <tr><td>Mean Confidence</td><td>{results['confidence_stats']['mean']:.3f}</td></tr>
                        <tr><td>Confidence Std</td><td>{results['confidence_stats']['std']:.3f}</td></tr>
                    </table>
                </div>
        """
        
        if results.get('spatial_stats'):
            html += f"""
                <div class="section">
                    <h2>Spatial Analysis</h2>
                    <table>
                        <tr><th>Metric</th><th>Value</th></tr>
                        <tr><td>Number of Clusters</td><td>{results['spatial_stats']['n_clusters']}</td></tr>
                        <tr><td>Cells in Clusters</td><td>{results['spatial_stats']['points_in_clusters']}</td></tr>
                        <tr><td>Isolated Cells</td><td>{results['spatial_stats']['isolated_points']}</td></tr>
                    </table>
                </div>
            """
            
        html += """
                <div class="section">
                    <h2>Visualizations</h2>
                    <div class="plot">
                        <img src="count_distribution.png" alt="Cell Count Distribution">
                        <p>Distribution of cell counts across tiles</p>
                    </div>
                    <div class="plot">
                        <img src="confidence_stats.png" alt="Confidence Statistics">
                        <p>Distribution of detection confidence scores</p>
                    </div>
        """
        
        if has_density_map:
            html += """
                    <div class="plot">
                        <img src="density_map.png" alt="Cell Density Map">
                        <p>Spatial distribution of detected cells</p>
                    </div>
            """
            
        if sample_images:
            html += """
                <div class="section">
                    <h2>Sample Images</h2>
                    <div class="sample-images">
            """
            for path, caption in sample_images:
                html += f"""
                        <div class="sample-image">
                            <img src="{path}" alt="Sample Image">
                            <p>{caption}</p>
                        </div>
                """
            html += "</div></div>"
            
        html += """
            </div>
        </body>
        </html>
        """
        
        return html 

    def generate_summary(self, results: Dict[str, Any], image: np.ndarray) -> str:
        """Generate a summary of the analysis results using MiniGPT-4."""
        # Convert image to PIL format
        image_pil = Image.fromarray(image)
        
        # Prepare prompt
        prompt = f"""
        Based on the following cell analysis results, provide a comprehensive summary:
        
        Analysis Results:
        {json.dumps(results, indent=2)}
        
        Please provide:
        1. A clear overview of the findings
        2. Key statistics and metrics
        3. Notable patterns or trends
        4. Biological significance
        """
        
        # Generate summary with MiniGPT-4
        self.minigpt4.upload_img(image_pil)
        self.minigpt4.ask(prompt)
        summary = self.minigpt4.answer()
        
        return summary
        
    def generate_visual_description(self, image: np.ndarray) -> str:
        """Generate a detailed visual description using BLIP-2."""
        # Convert image to PIL format
        image_pil = Image.fromarray(image)
        
        # Generate description with BLIP-2
        inputs = self.blip2_processor(image_pil, return_tensors="pt")
        out = self.blip2_model.generate(**inputs)
        description = self.blip2_processor.decode(out[0], skip_special_tokens=True)
        
        return description
        
    def analyze_patterns(self, results: Dict[str, Any], image: np.ndarray) -> str:
        """Analyze patterns in the results using LLaVA."""
        # Convert image to PIL format
        image_pil = Image.fromarray(image)
        
        # Prepare prompt
        prompt = f"""
        Analyze this microscopy image and the following results for significant patterns:
        
        Results:
        {json.dumps(results, indent=2)}
        
        Please identify and explain:
        1. Spatial patterns in cell distribution
        2. Any clustering or organization
        3. Potential biological implications
        4. Areas requiring further investigation
        """
        
        # Get LLaVA analysis
        analysis = eval_model(
            self.llava_model,
            self.llava_tokenizer,
            self.llava_image_processor,
            prompt,
            image_pil,
            self.llava_context_len
        )
        
        return analysis
        
    def generate_recommendations(self, results: Dict[str, Any]) -> str:
        """Generate recommendations using Mistral."""
        # Prepare prompt
        prompt = f"""
        Based on the following cell analysis results, provide recommendations:
        
        Results:
        {json.dumps(results, indent=2)}
        
        Please provide:
        1. Suggestions for further analysis
        2. Potential follow-up experiments
        3. Areas requiring additional investigation
        4. Technical recommendations for improvement
        """
        
        # Generate recommendations with Mistral
        inputs = self.mistral_tokenizer(prompt, return_tensors="pt")
        outputs = self.mistral_model.generate(**inputs, max_length=500)
        recommendations = self.mistral_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return recommendations
        
    def create_report(self, results: Dict[str, Any], image: np.ndarray, 
                     output_dir: str) -> Dict[str, str]:
        """Create a comprehensive report with multiple sections."""
        # Generate report sections
        summary = self.generate_summary(results, image)
        visual_description = self.generate_visual_description(image)
        pattern_analysis = self.analyze_patterns(results, image)
        recommendations = self.generate_recommendations(results)
        
        # Create report dictionary
        report = {
            'summary': summary,
            'visual_description': visual_description,
            'pattern_analysis': pattern_analysis,
            'recommendations': recommendations
        }
        
        # Save report to files
        os.makedirs(output_dir, exist_ok=True)
        
        # Save JSON report
        with open(os.path.join(output_dir, 'report.json'), 'w') as f:
            json.dump(report, f, indent=2)
            
        # Save text report
        with open(os.path.join(output_dir, 'report.txt'), 'w') as f:
            f.write("Cell Analysis Report\n")
            f.write("===================\n\n")
            f.write("Summary\n")
            f.write("-------\n")
            f.write(summary + "\n\n")
            f.write("Visual Description\n")
            f.write("-----------------\n")
            f.write(visual_description + "\n\n")
            f.write("Pattern Analysis\n")
            f.write("---------------\n")
            f.write(pattern_analysis + "\n\n")
            f.write("Recommendations\n")
            f.write("--------------\n")
            f.write(recommendations + "\n")
            
        return report 