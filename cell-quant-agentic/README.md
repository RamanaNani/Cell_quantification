# Cell Quantification Pipeline

An agent-based pipeline for analyzing and quantifying cells in microscopy images.

## Features

- Load and process VSI microscopy files
- Extract and analyze regions around markers
- Detect and quantify cells using deep learning
- Generate comprehensive analysis reports
- Interactive web interface using Streamlit

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/cell-quant-agentic.git
cd cell-quant-agentic
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install system dependencies:
- OpenSlide: Follow instructions at [OpenSlide website](https://openslide.org/download/)
- Java Development Kit (JDK): Required for Bio-Formats

## Usage

### Command Line Interface

Run the pipeline from the command line:

```bash
python main_pipeline.py path/to/image.vsi path/to/markers.xml --output results
```

Optional arguments:
- `--output`, `-o`: Output directory (default: results)
- `--tile-size`, `-t`: Size of extracted tiles (default: 512)
- `--confidence`, `-c`: Detection confidence threshold (default: 0.5)
- `--config`: Path to configuration JSON file

### Web Interface

Launch the Streamlit web interface:

```bash
streamlit run app/streamlit_app.py
```

Then open your browser and navigate to the displayed URL (usually http://localhost:8501).

## Project Structure

```
cell-quant-agentic/
├── agents/                 # Agent modules
│   ├── file_handler.py    # VSI and XML file handling
│   ├── preprocessing.py   # Image preprocessing
│   ├── detection.py       # Cell detection
│   ├── quantification.py  # Analysis and statistics
│   ├── report.py         # Report generation
│   └── chat_agent.py     # User interaction
├── app/
│   └── streamlit_app.py  # Web interface
├── models/               # Model weights and configs
├── data/                # Data directory
│   ├── raw/            # Raw input files
│   └── tiles/          # Extracted image tiles
├── results/            # Analysis results
├── utils/             # Utility functions
├── main_pipeline.py   # Main pipeline script
├── README.md         # This file
└── requirements.txt  # Python dependencies
```

## Configuration

Create a JSON configuration file to customize pipeline parameters:

```json
{
    "tile_size": 512,
    "confidence_threshold": 0.5,
    "model_path": "models/cell_detector.pth",
    "preprocessing": {
        "normalize": true,
        "enhance_contrast": true
    }
}
```

## Output

The pipeline generates:
- Extracted image tiles
- Preprocessed images
- Detection visualizations
- Analysis reports (JSON, CSV, HTML)
- Cell density heatmaps
- Summary statistics

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenSlide for VSI file handling
- Bio-Formats for microscopy file support
- PyTorch for deep learning capabilities 