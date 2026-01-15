# Russian License Plate Detection and Recognition

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)](https://opencv.org/)

## Project Overview

This project implements an automated system for detecting and recognizing Russian license plates in images using computer vision techniques, cascade classifiers, and OCR (Optical Character Recognition).

## Features

- **License Plate Detection**: Uses Haar Cascade classifiers trained specifically for Russian license plates
- **Automatic Alignment**: Detects and corrects plate rotation using Hough Line Transform
- **Image Preprocessing**: Multiple preprocessing levels (light, medium, heavy) with CLAHE, bilateral filtering, and adaptive thresholding
- **Character Recognition**: Tesseract OCR with multiple PSM modes for optimal accuracy
- **Format Validation**: Validates detected text against Russian license plate format (Letter + 3 digits + 2 letters)
- **Comprehensive Visualization**: Saves annotated images, extracted plates, and processing steps
- **Detailed Logging**: JSON output with detection results, confidence scores, and processing metrics

## Requirements

### System Dependencies
- Python 3.8+
- Tesseract OCR 4.0+

### Python Libraries (included in venv)
- opencv-python
- numpy
- matplotlib
- pytesseract

## Installation

The project comes with a pre-configured virtual environment. If you need to recreate it:

```bash
cd "final project"
python3 -m venv venv
source venv/bin/activate
pip install opencv-python numpy matplotlib pytesseract
```

Ensure Tesseract OCR is installed on your system:
- macOS: `brew install tesseract`
- Ubuntu: `sudo apt-get install tesseract-ocr`
- Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki

## Usage

### Running the Detection

```bash
./run_detection.sh
```

Or manually:

```bash
source venv/bin/activate
python license_plate_detection.py
```

### Input Images

Place test images in the `input_images/` directory. The script processes:
- Russia close.jpg
- Russia Far.png
- Not Russian.jpeg

### Output Structure

Results are saved in the `output/` directory:

```
output/
├── annotated_originals/      # Original images with red bounding boxes
│   ├── Russia close.jpg_annotated.jpg
│   ├── Russia Far.png_annotated.jpg
│   └── Not Russian.jpeg_annotated.jpg
├── detected_plates/           # Extracted plate regions
│   ├── *_plate_1_raw.jpg     # Raw extracted plates
│   └── *_plate_1_aligned.jpg # Rotation-corrected plates
├── processed_plates/          # Preprocessed plates ready for OCR
│   └── *_plate_1_processed.jpg
└── results.json               # Detailed processing results and metrics
```

## Algorithm Overview

### Phase 1: Detection
1. Load and preprocess image (grayscale conversion, CLAHE enhancement)
2. Apply Russian plate cascade classifier
3. Validate detections based on aspect ratio (2:1 to 6:1) and size
4. Extract valid plate regions with padding

### Phase 2: Alignment
1. Detect edges using Canny edge detection
2. Find dominant lines using Hough Line Transform
3. Calculate rotation angle from line angles
4. Apply affine transformation to correct rotation

### Phase 3: OCR Preprocessing
1. Scale plate image (minimum 100px height, 3x upscale)
2. Apply CLAHE for contrast enhancement
3. Bilateral filtering to preserve edges
4. Otsu's automatic thresholding
5. Morphological operations (opening and closing)
6. Invert if background is dark

### Phase 4: Character Recognition
1. Try multiple Tesseract PSM modes (7, 8, 6, 13)
2. Whitelist alphanumeric characters only
3. Select best result based on confidence and length
4. Validate against Russian plate format: Letter + 3 digits + 2 letters

## Key Parameters

### Detection Parameters
- `scaleFactor`: 1.1 (how much image is scaled down at each scale)
- `minNeighbors`: 3 (minimum neighbors for valid detection)
- `minSize`: (30, 10) (minimum plate size in pixels)

### Preprocessing Levels
- **none**: Raw grayscale conversion
- **light**: Histogram equalization
- **medium**: CLAHE + Gaussian blur (default)
- **heavy**: CLAHE + bilateral filtering

### OCR Configuration
- Multiple PSM modes for robustness
- Character whitelist: A-Z, 0-9
- Confidence thresholding
- Format validation

## Results

### Iteration 1 - Baseline Detection

**Detection Results:**
- Total images processed: 3
- Images with detections: 3
- Total plates detected: 4

**Performance:**
- Not Russian.jpeg: 1 plate detected (0.34s) - Recognized: 6FFGWZ
- Russia Far.png: 2 plates detected (0.75s) - Recognized: 32707 (partial)
- Russia close.jpg: 1 plate detected (0.38s) - Recognized: 00T77 (partial)

**Observations:**
1. Cascade classifier successfully detects plates in all images
2. Russian classifier did detect the non-Russian plate (may need parameter tuning)
3. OCR recognizes characters but accuracy needs improvement
4. Far distance detection works but character recognition is challenging
5. Preprocessing and OCR improvements needed for better character accuracy

## Challenges and Solutions

### Challenge 1: Small Plates in Far Images
- **Solution**: Minimum size parameter set to (30, 10), upscaling in OCR preprocessing

### Challenge 2: OCR Accuracy
- **Solution**: Multiple PSM modes, aggressive preprocessing, Otsu's thresholding, 3x upscaling

### Challenge 3: Varying Illumination
- **Solution**: CLAHE for adaptive contrast enhancement, bilateral filtering

### Challenge 4: Plate Rotation
- **Solution**: Hough Transform for angle detection, affine transformation for correction

### Challenge 5: Non-Russian Plate Detection
- **Solution**: Aspect ratio validation, format validation (though baseline detected it)

## Future Improvements

1. **Enhanced OCR Preprocessing**: Try additional preprocessing techniques like sharpening, deskewing
2. **Post-processing**: Implement character-level validation and correction
3. **Deep Learning**: Consider CNN-based plate detection and recognition for higher accuracy
4. **Multi-iteration Testing**: Run multiple parameter combinations and compare results
5. **Character Segmentation**: Segment individual characters for better recognition
6. **Training Data**: Fine-tune Tesseract with Russian plate character patterns
7. **False Positive Filtering**: Improve validation to reduce non-Russian plate detection

## File Structure

```
final project/
├── license_plate_detection.py  # Main executable Python script
├── run_detection.sh            # Convenience wrapper script
├── PLANNING.md                 # Detailed project plan
├── README.md                   # This file
├── cascades/                   # Cascade classifier XML files
│   ├── haarcascade_russian_plate_number.xml
│   └── haarcascade_license_plate_rus_16stages.xml
├── input_images/               # Test images
│   ├── Russia close.jpg
│   ├── Russia Far.png
│   └── Not Russian.jpeg
├── output/                     # Results directory
└── venv/                       # Python virtual environment
```

## Code Documentation

The main script is well-documented with:
- Class and method docstrings
- Parameter descriptions
- Return value specifications
- Inline comments for complex operations

### Key Functions

1. `load_and_preprocess()`: Load and preprocess images
2. `detect_plates()`: Detect plates using cascade classifier
3. `validate_plate_detection()`: Validate based on aspect ratio
4. `extract_plate_region()`: Extract plate with padding
5. `align_plate()`: Rotate plate to horizontal
6. `process_plate_for_ocr()`: Preprocessing for OCR
7. `recognize_characters()`: OCR with Tesseract
8. `validate_plate_format()`: Validate Russian plate format
9. `draw_results()`: Visualize detections
10. `process_image()`: Complete pipeline for one image

## License

Educational project for computer vision and license plate recognition.

## Author

Created for CSU Global Computer Vision Course - Final Project
