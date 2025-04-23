#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
from typing import Dict, Any
import json
from datetime import datetime

from agents.file_handler import FileHandlerAgent
from agents.preprocessing import PreprocessingAgent
from agents.detection import DetectionAgent
from agents.quantification import QuantificationAgent
from agents.report import ReportAgent
from agents.chat_agent import ChatAgent

class Pipeline:
    """Main pipeline for cell quantification."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the pipeline.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.setup_agents()
        
    def setup_agents(self):
        """Initialize all agents."""
        self.file_handler = FileHandlerAgent()
        self.preprocessor = PreprocessingAgent()
        self.detector = DetectionAgent()
        self.quantifier = QuantificationAgent()
        self.report_generator = ReportAgent()
        self.chat_agent = ChatAgent()
        
    def run(self, vsi_path: str, xml_path: str, output_dir: str = "results",
            tile_size: int = 512, confidence_threshold: float = 0.5) -> Dict[str, Any]:
        """
        Run the complete analysis pipeline.
        
        Args:
            vsi_path: Path to VSI file
            xml_path: Path to XML file
            output_dir: Directory to save results
            tile_size: Size of tiles to extract
            confidence_threshold: Confidence threshold for detection
            
        Returns:
            Dict: Analysis results
        """
        # Start chat session
        session = self.chat_agent.start_session()
        
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Load files
            self.file_handler.load_slide(vsi_path)
            markers = self.file_handler.load_markers(xml_path)
            print(f"Found {len(markers)} markers in the XML file")
            
            # Get metadata
            meta = self.file_handler.get_metadata()
            print("\nSlide Metadata:")
            for k, v in meta.items():
                if k != "properties":
                    print(f"{k}: {v}")
                    
            # Extract tiles
            tiles_dir = os.path.join(output_dir, "tiles")
            tiles = self.file_handler.extract_tiles(
                markers,
                size=(tile_size, tile_size),
                output_dir=tiles_dir
            )
            print(f"\nExtracted {len(tiles)} tiles")
            
            # Preprocess tiles
            preprocessed_dir = os.path.join(output_dir, "preprocessed")
            preprocessed = self.preprocessor.preprocess_batch(tiles, preprocessed_dir)
            print("\nPreprocessing complete")
            
            # Detect cells
            detection_dir = os.path.join(output_dir, "detections")
            detections = self.detector.process_batch(preprocessed, detection_dir)
            print("\nCell detection complete")
            
            # Analyze results
            analysis = self.quantifier.analyze_detections(
                detections,
                tile_coords=[(m[0], m[1]) for m in markers]
            )
            
            # Generate density map
            density_map = self.quantifier.calculate_density_map(
                detections,
                [(m[0], m[1]) for m in markers],
                meta['dimensions']
            )
            
            # Generate report
            self.report_generator.create_report(
                analysis,
                density_map=density_map,
                sample_images=[(p, f"Tile {i+1}") for i, p in enumerate(tiles[:4])],
                output_dir=output_dir
            )
            print(f"\nResults saved in {output_dir}/")
            
            # Update session
            session['analysis_results'] = analysis
            self.chat_agent.save_session(output_dir)
            
            return analysis
            
        except Exception as e:
            print(f"Error: {str(e)}")
            raise
            
def main():
    parser = argparse.ArgumentParser(description="Cell Quantification Pipeline")
    parser.add_argument("vsi_path", help="Path to the VSI file")
    parser.add_argument("xml_path", help="Path to the XML file with markers")
    parser.add_argument("--output", "-o", default="results",
                       help="Output directory (default: results)")
    parser.add_argument("--tile-size", "-t", type=int, default=512,
                       help="Tile size for extraction (default: 512)")
    parser.add_argument("--confidence", "-c", type=float, default=0.5,
                       help="Confidence threshold for detection (default: 0.5)")
    parser.add_argument("--config", type=str,
                       help="Path to configuration JSON file")
    
    args = parser.parse_args()
    
    # Load configuration if provided
    config = None
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
            
    # Run pipeline
    pipeline = Pipeline(config)
    pipeline.run(
        args.vsi_path,
        args.xml_path,
        args.output,
        args.tile_size,
        args.confidence
    )

if __name__ == "__main__":
    main() 