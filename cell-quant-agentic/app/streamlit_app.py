import streamlit as st
import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import pandas as pd

# Add parent directory to path to import agents
sys.path.append(str(Path(__file__).parent.parent))
from agents.file_handler import FileHandlerAgent
from agents.preprocessing import PreprocessingAgent
from agents.detection import DetectionAgent
from agents.quantification import QuantificationAgent
from agents.report import ReportAgent

def main():
    st.set_page_config(
        page_title="Cell Quantification Pipeline",
        page_icon="🔬",
        layout="wide"
    )
    
    st.title("Cell Quantification Pipeline")
    
    # Sidebar
    st.sidebar.title("Settings")
    
    # File Upload
    st.sidebar.header("File Upload")
    vsi_file = st.sidebar.file_uploader("Upload VSI File", type=["vsi"])
    xml_file = st.sidebar.file_uploader("Upload XML File", type=["xml"])
    
    # Analysis Parameters
    st.sidebar.header("Analysis Parameters")
    tile_size = st.sidebar.slider("Tile Size", 256, 1024, 512, 128)
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.1)
    
    # Main content
    if vsi_file is None or xml_file is None:
        st.info("Please upload both VSI and XML files to begin analysis.")
        return
        
    # Save uploaded files
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    
    vsi_path = os.path.join(temp_dir, vsi_file.name)
    xml_path = os.path.join(temp_dir, xml_file.name)
    
    with open(vsi_path, "wb") as f:
        f.write(vsi_file.getbuffer())
    with open(xml_path, "wb") as f:
        f.write(xml_file.getbuffer())
        
    # Initialize agents
    file_handler = FileHandlerAgent(vsi_path, xml_path)
    preprocessor = PreprocessingAgent()
    detector = DetectionAgent()
    quantifier = QuantificationAgent()
    report_generator = ReportAgent()
    
    # Process pipeline
    try:
        # Load slide and markers
        with st.spinner("Loading files..."):
            file_handler.load_slide()
            markers = file_handler.load_markers()
            st.success(f"Found {len(markers)} markers in the XML file")
            
            # Show metadata
            meta = file_handler.get_metadata()
            st.subheader("Slide Metadata")
            st.json(meta)
            
        # Extract and process tiles
        with st.spinner("Processing tiles..."):
            tiles = file_handler.extract_tiles(markers, size=(tile_size, tile_size))
            st.success(f"Extracted {len(tiles)} tiles")
            
            # Show sample tiles
            st.subheader("Sample Tiles")
            cols = st.columns(4)
            for i, tile_path in enumerate(tiles[:4]):
                cols[i].image(tile_path, caption=f"Tile {i+1}")
                
        # Preprocess tiles
        with st.spinner("Preprocessing tiles..."):
            preprocessed = preprocessor.preprocess_batch(tiles)
            st.success("Preprocessing complete")
            
        # Detect cells
        with st.spinner("Detecting cells..."):
            detections = detector.process_batch(preprocessed)
            
            # Show sample detections
            st.subheader("Sample Detections")
            cols = st.columns(4)
            for i, (image, dets) in enumerate(zip(preprocessed[:4], detections[:4])):
                vis_image = detector.visualize_detections(image, dets)
                cols[i].image(vis_image, caption=f"Detections in Tile {i+1}")
                
        # Analyze results
        with st.spinner("Analyzing results..."):
            analysis = quantifier.analyze_detections(detections)
            
            # Show analysis results
            st.subheader("Analysis Results")
            st.json(analysis)
            
            # Generate density map
            density_map = quantifier.calculate_density_map(
                detections, markers, meta['dimensions']
            )
            
            st.subheader("Cell Density Map")
            st.image(density_map, caption="Spatial distribution of detected cells")
            
        # Generate report
        with st.spinner("Generating report..."):
            report_generator.create_report(
                analysis,
                density_map=density_map,
                sample_images=[(tile_path, f"Tile {i+1}") for i, tile_path in enumerate(tiles[:4])]
            )
            st.success("Report generated successfully")
            
    except Exception as e:
        st.error(f"Error during processing: {str(e)}")
        
    finally:
        # Cleanup
        if os.path.exists(vsi_path):
            os.remove(vsi_path)
        if os.path.exists(xml_path):
            os.remove(xml_path)

if __name__ == "__main__":
    main() 