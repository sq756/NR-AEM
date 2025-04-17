# NR-AEM Neutron Imaging Analysis Tool

## Project Description
This tool is designed for analyzing two-phase flow in AEM (Anion Exchange Membrane) images using neutron radiography. It provides comprehensive visualization and quantitative analysis capabilities for 16-bit TIFF images obtained from neutron imaging experiments.

## Key Features
- Interactive GUI for visualizing 16-bit TIFF images with adjustable intensity ranges
- Background subtraction and denoising (mean/median filters)
- Edge detection with adjustable thresholds
- Region masking and analysis tools
- Pseudo-color visualization with customizable color maps
- Real-time histogram analysis
- Multiple image comparison (original, denoised, background subtracted)
- Save processed images in various formats

## Installation
1. Python 3.7+ required
2. Install dependencies:
```
pip install numpy pillow matplotlib scipy scikit-image opencv-python-headless tk
```

## Usage
Run the main script:
```
python opencv.py
```

### Main Functions
- **Open 16-bit TIFF**: Load neutron radiography images
- **Load Background**: Subtract background for better contrast
- **Apply Denoising**: Reduce noise using mean or median filters
- **Edge Detection**: Identify phase boundaries
- **Analyze Region**: Quantify selected regions
- **Save Images**: Export processed images

## Scientific Applications
- Quantitative analysis of two-phase flow in AEM
- Water content distribution mapping
- Phase boundary detection
- Neutron attenuation measurements

## License
MIT License
