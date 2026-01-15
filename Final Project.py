#!/usr/bin/env python3
"""
License Plate Detection and Character Recognition
Russian License Plate Detection using Cascade Classifiers and OCR

This program detects Russian license plates in images and performs character recognition.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import time
import re
from datetime import datetime

# Try to import pytesseract for OCR
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("Warning: pytesseract not available. Character recognition will be limited.")

# Try to import EasyOCR
try:
    import easyocr
    EASYOCR_AVAILABLE = True
    EASYOCR_READER = None  # Will be initialized on first use
except ImportError:
    EASYOCR_AVAILABLE = False
    print("Warning: easyocr not available. Falling back to Tesseract only.")


class LicensePlateDetector:
    """Main class for license plate detection and recognition"""

    def __init__(self, cascade_path, char_cascade_path=None):
        """
        Initialize the detector with cascade classifier paths

        Args:
            cascade_path: Path to Russian plate cascade classifier
            char_cascade_path: Path to character/number cascade classifier (optional)
        """
        self.plate_cascade = cv2.CascadeClassifier(cascade_path)
        self.char_cascade = None
        if char_cascade_path and os.path.exists(char_cascade_path):
            self.char_cascade = cv2.CascadeClassifier(char_cascade_path)

        if self.plate_cascade.empty():
            raise ValueError(f"Failed to load cascade classifier from {cascade_path}")

        self.results = []
        self.log = []

    def log_message(self, message):
        """Log a message with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.log.append(log_entry)
        print(log_entry)

    def load_and_preprocess(self, image_path, preprocessing_level='medium'):
        """
        Load image and apply preprocessing

        Args:
            image_path: Path to input image
            preprocessing_level: 'none', 'light', 'medium', or 'heavy'

        Returns:
            original: Original color image
            gray: Grayscale image
            processed: Preprocessed grayscale image
        """
        self.log_message(f"Loading image: {os.path.basename(image_path)}")

        # Load original image
        original = cv2.imread(image_path)
        if original is None:
            raise ValueError(f"Failed to load image from {image_path}")

        # Convert to grayscale
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

        # Apply preprocessing based on level
        if preprocessing_level == 'none':
            processed = gray.copy()
        elif preprocessing_level == 'light':
            # Simple histogram equalization
            processed = cv2.equalizeHist(gray)
        elif preprocessing_level == 'medium':
            # CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            processed = clahe.apply(gray)
            # Slight blur to reduce noise
            processed = cv2.GaussianBlur(processed, (3, 3), 0)
        elif preprocessing_level == 'heavy':
            # CLAHE + bilateral filtering
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            processed = clahe.apply(gray)
            processed = cv2.bilateralFilter(processed, 9, 75, 75)
        else:
            processed = gray.copy()

        return original, gray, processed

    def validate_plate_detection(self, bbox, image_shape, plate_region=None):
        """
        Validate detected plate based on aspect ratio, size, and color

        Args:
            bbox: Bounding box tuple (x, y, w, h)
            image_shape: Shape of the image (height, width)
            plate_region: Optional plate image region for color validation

        Returns:
            Boolean indicating if detection is valid
        """
        x, y, w, h = bbox
        aspect_ratio = w / h if h > 0 else 0

        # Russian plates typically have aspect ratio between 2.5:1 and 4.5:1 (tighter range)
        if aspect_ratio < 2.5 or aspect_ratio > 4.5:
            self.log_message(f"Rejected: aspect ratio {aspect_ratio:.2f} outside range [2.5, 4.5]")
            return False

        # Check if size is reasonable (not too small, not too large)
        image_h, image_w = image_shape[:2]
        area = w * h
        image_area = image_h * image_w

        if area < (image_area * 0.001):  # Too small
            self.log_message(f"Rejected: area too small ({area} < {image_area * 0.001:.0f})")
            return False
        if area > (image_area * 0.3):  # Too large
            self.log_message(f"Rejected: area too large ({area} > {image_area * 0.3:.0f})")
            return False

        # Position heuristic: plates typically in lower 2/3 of image
        if y < image_h * 0.2:  # Too high in image
            self.log_message(f"Rejected: position too high (y={y} < {image_h * 0.2:.0f})")
            return False

        # Color validation if plate region provided
        if plate_region is not None:
            if not self.validate_plate_color(plate_region):
                self.log_message(f"Rejected: color validation failed")
                return False

        return True

    def validate_plate_color(self, plate_region):
        """
        Validate that plate has characteristics of Russian license plate
        Russian plates have white background with black text
        Made more permissive to avoid rejecting valid plates

        Args:
            plate_region: BGR image of the plate region

        Returns:
            Boolean indicating if color pattern matches Russian plate
        """
        if len(plate_region.shape) != 3:
            return True  # Skip if grayscale

        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(plate_region, cv2.COLOR_BGR2HSV)

        # Russian plates are predominantly white/light colored
        # Check if majority of pixels are bright (high V value)
        v_channel = hsv[:, :, 2]
        bright_pixels = np.sum(v_channel > 150)  # Lowered from 180 to be more permissive
        total_pixels = v_channel.size
        bright_ratio = bright_pixels / total_pixels

        # Should have at least 25% bright pixels (white background) - lowered from 40%
        if bright_ratio < 0.25:
            self.log_message(f"Color check: bright_ratio={bright_ratio:.2f} < 0.25")
            return False

        # Check for relatively low saturation (white/gray colors)
        s_channel = hsv[:, :, 1]
        low_saturation = np.sum(s_channel < 80)  # Increased threshold from 60
        low_sat_ratio = low_saturation / total_pixels

        # Should have at least 35% low saturation pixels - lowered from 50%
        if low_sat_ratio < 0.35:
            self.log_message(f"Color check: low_sat_ratio={low_sat_ratio:.2f} < 0.35")
            return False

        self.log_message(f"Color check passed: bright={bright_ratio:.2f}, low_sat={low_sat_ratio:.2f}")
        return True

    def detect_plates(self, image, gray_image, params=None):
        """
        Detect license plates using cascade classifier

        Args:
            image: Original color image
            gray_image: Preprocessed grayscale image
            params: Dictionary with detection parameters

        Returns:
            List of validated bounding boxes
        """
        if params is None:
            params = {
                'scaleFactor': 1.1,
                'minNeighbors': 3,
                'minSize': (50, 15)
            }

        self.log_message(f"Detecting plates with params: {params}")

        # Detect plates
        plates = self.plate_cascade.detectMultiScale(
            gray_image,
            scaleFactor=params.get('scaleFactor', 1.1),
            minNeighbors=params.get('minNeighbors', 3),
            minSize=params.get('minSize', (50, 15))
        )

        self.log_message(f"Initial detections: {len(plates)}")

        # Validate detections with color checking
        valid_plates = []
        for i, (x, y, w, h) in enumerate(plates):
            # Extract plate region for color validation
            plate_region = self.extract_plate_region(image, (x, y, w, h), padding=0)

            if self.validate_plate_detection((x, y, w, h), image.shape, plate_region):
                valid_plates.append((x, y, w, h))
                self.log_message(f"Detection {i+1} ACCEPTED: bbox=[{x},{y},{w},{h}], aspect={w/h:.2f}")
            else:
                self.log_message(f"Detection {i+1} REJECTED: bbox=[{x},{y},{w},{h}], aspect={w/h:.2f}")

        self.log_message(f"Valid detections after filtering: {len(valid_plates)}")

        return valid_plates

    def extract_plate_region(self, image, bbox, padding=5):
        """
        Extract plate region from image with optional padding

        Args:
            image: Input image
            bbox: Bounding box tuple (x, y, w, h)
            padding: Padding in pixels

        Returns:
            Extracted plate image
        """
        x, y, w, h = bbox

        # Add padding
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(image.shape[1], x + w + padding)
        y_end = min(image.shape[0], y + h + padding)

        plate_img = image[y_start:y_end, x_start:x_end]
        return plate_img

    def align_plate(self, plate_image):
        """
        Rotate plate to horizontal alignment if needed

        Args:
            plate_image: Input plate image

        Returns:
            Aligned plate image, rotation angle
        """
        # Convert to grayscale if needed
        if len(plate_image.shape) == 3:
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_image

        # Detect edges
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Detect lines using Hough Transform
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=50)

        if lines is None or len(lines) == 0:
            return plate_image, 0.0

        # Calculate median angle of detected lines
        angles = []
        for line in lines:
            rho, theta = line[0]
            angle = theta * 180 / np.pi
            # Normalize angle to [-45, 45]
            if angle > 135:
                angle = angle - 180
            elif angle > 45:
                angle = angle - 90
            if abs(angle) < 45:
                angles.append(angle)

        if len(angles) == 0:
            return plate_image, 0.0

        rotation_angle = np.median(angles)

        # Only rotate if angle is significant
        if abs(rotation_angle) < 2.0:
            return plate_image, rotation_angle

        # Rotate image
        (h, w) = plate_image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
        rotated = cv2.warpAffine(plate_image, M, (w, h),
                                 flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)

        return rotated, rotation_angle

    def process_plate_for_ocr(self, plate_image, iteration='adaptive'):
        """
        Apply preprocessing for optimal character recognition

        Args:
            plate_image: Input plate image
            iteration: Preprocessing strategy ('adaptive', 'standard', 'aggressive')

        Returns:
            Processed plate image ready for OCR
        """
        # Convert to grayscale if needed
        if len(plate_image.shape) == 3:
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_image

        # Adaptive upscaling based on plate size
        h, w = gray.shape
        if h < 30:
            scale_factor = 6  # Very small plates (far away)
        elif h < 50:
            scale_factor = 5  # Small plates
        elif h < 80:
            scale_factor = 4  # Medium plates
        else:
            scale_factor = 3  # Large plates (close up)

        self.log_message(f"Plate size: {w}x{h}, upscaling by {scale_factor}x")

        new_h = int(h * scale_factor)
        new_w = int(w * scale_factor)
        resized = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        # Denoise first
        denoised = cv2.fastNlMeansDenoising(resized, None, 10, 7, 21)

        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

        # Apply sharpening filter
        kernel_sharpen = np.array([[-1, -1, -1],
                                   [-1,  9, -1],
                                   [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel_sharpen)

        # Apply bilateral filter to preserve edges while reducing noise
        filtered = cv2.bilateralFilter(sharpened, 9, 75, 75)

        # Try multiple thresholding methods and keep best
        # Method 1: Otsu's thresholding
        _, binary1 = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Method 2: Inverted Otsu's
        _, binary2 = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Method 3: Adaptive thresholding
        binary3 = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)

        # Choose the one with more reasonable character-like regions
        # For now, use standard binary (will enhance with voting later)
        binary = binary1

        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        # Remove noise
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        # Close gaps
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)

        return cleaned

    def segment_characters(self, binary_plate):
        """
        Segment individual characters from plate using contour detection

        Args:
            binary_plate: Binary/thresholded plate image

        Returns:
            List of character bounding boxes sorted left to right
        """
        # Try both normal and inverted for better character detection
        # Sometimes characters are white on black, sometimes black on white
        binary_normal = binary_plate.copy()
        binary_inverted = cv2.bitwise_not(binary_plate)

        best_chars = []
        best_count = 0

        for binary_version in [binary_normal, binary_inverted]:
            # Find contours
            contours, _ = cv2.findContours(binary_version, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

            # Filter contours for character-like shapes
            char_candidates = []
            h, w = binary_version.shape
            plate_area = h * w

            for cnt in contours:
                x, y, width, height = cv2.boundingRect(cnt)
                aspect_ratio = height / width if width > 0 else 0
                area = width * height

                # More permissive character validation rules
                # Russian plate characters can vary in aspect ratio
                if (0.8 < aspect_ratio < 5.0 and           # Wider range for aspect ratio
                    area > (plate_area * 0.002) and        # Lower threshold for small chars
                    area < (plate_area * 0.4) and          # Allow slightly larger
                    height > (h * 0.2) and                 # Lower height threshold
                    height < (h * 0.95) and                # Allow taller
                    width > (w * 0.01)):                   # Minimum width
                    char_candidates.append((x, y, width, height))

            # Sort left to right
            char_candidates.sort(key=lambda c: c[0])

            # Filter overlapping/duplicate detections
            filtered_chars = []
            for i, char in enumerate(char_candidates):
                x, y, w, h = char
                # Check if overlaps significantly with previous
                is_duplicate = False
                for prev_x, prev_y, prev_w, prev_h in filtered_chars:
                    # Calculate overlap
                    overlap_x = max(0, min(x + w, prev_x + prev_w) - max(x, prev_x))
                    if overlap_x > w * 0.5:  # More than 50% overlap
                        is_duplicate = True
                        break

                if not is_duplicate:
                    filtered_chars.append(char)

            # Keep the version with most characters found
            if len(filtered_chars) > best_count:
                best_chars = filtered_chars
                best_count = len(filtered_chars)

        self.log_message(f"Character segmentation: found {len(best_chars)} candidates")

        # Validate we have reasonable number of characters (6-8 for Russian plates)
        if len(best_chars) < 4 or len(best_chars) > 10:
            self.log_message(f"WARNING: Unusual character count: {len(best_chars)}")

        return best_chars

    def recognize_characters_segmented(self, plate_image, binary_plate):
        """
        Recognize characters using segmentation approach

        Args:
            plate_image: Original grayscale plate image
            binary_plate: Binary version for segmentation

        Returns:
            Recognized text, character bounding boxes, confidence scores
        """
        # Segment characters
        char_regions = self.segment_characters(binary_plate)

        if len(char_regions) == 0:
            self.log_message("No characters segmented, falling back to whole-plate OCR")
            return None, [], []

        # OCR each character individually
        recognized_chars = []
        confidences = []
        char_boxes = []

        for i, (x, y, w, h) in enumerate(char_regions):
            # Extract character with padding
            padding = 3
            x_start = max(0, x - padding)
            y_start = max(0, y - padding)
            x_end = min(plate_image.shape[1], x + w + padding)
            y_end = min(plate_image.shape[0], y + h + padding)

            char_img = plate_image[y_start:y_end, x_start:x_end]

            if char_img.size == 0:
                continue

            # Upscale significantly (individual characters are small)
            char_h, char_w = char_img.shape
            target_height = 128
            target_width = int(char_w * (target_height / char_h))
            char_img_large = cv2.resize(char_img, (target_width, target_height),
                                       interpolation=cv2.INTER_CUBIC)

            # Apply additional preprocessing for single character
            # Ensure white background
            if np.mean(char_img_large) < 127:
                char_img_large = cv2.bitwise_not(char_img_large)

            # OCR single character (PSM 10 = single character)
            custom_config = r'--oem 3 --psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

            try:
                # Get character
                char_text = pytesseract.image_to_string(char_img_large, config=custom_config).strip()

                # Get confidence
                data = pytesseract.image_to_data(char_img_large, config=custom_config,
                                                output_type=pytesseract.Output.DICT)
                char_conf = 0.0
                for j, text in enumerate(data['text']):
                    if text.strip():
                        char_conf = float(data['conf'][j]) / 100.0
                        break

                # Clean character
                char_text = re.sub(r'[^A-Z0-9]', '', char_text.upper())

                if char_text:
                    # Apply position-based correction
                    corrected_char = self.correct_character_by_position(char_text[0], i)
                    recognized_chars.append(corrected_char)
                    confidences.append(char_conf)
                    char_boxes.append((x, y, w, h))

                    self.log_message(f"Char {i}: '{char_text}' -> '{corrected_char}' (conf: {char_conf:.2%})")
                else:
                    self.log_message(f"Char {i}: No text detected")

            except Exception as e:
                self.log_message(f"Error OCR char {i}: {str(e)}")
                continue

        # Combine characters
        final_text = ''.join(recognized_chars)
        avg_confidence = np.mean(confidences) if confidences else 0.0

        self.log_message(f"Segmented OCR result: '{final_text}' (avg conf: {avg_confidence:.2%})")

        return final_text, char_boxes, confidences

    def get_easyocr_reader(self):
        """
        Get or initialize EasyOCR reader (singleton pattern to avoid reloading)

        Returns:
            EasyOCR Reader instance or None if not available
        """
        global EASYOCR_READER

        if not EASYOCR_AVAILABLE:
            return None

        if EASYOCR_READER is None:
            self.log_message("Initializing EasyOCR reader (first use only)...")
            try:
                # Initialize with English language, no GPU (for compatibility)
                EASYOCR_READER = easyocr.Reader(['en'], gpu=False)
                self.log_message("EasyOCR reader initialized successfully")
            except Exception as e:
                self.log_message(f"ERROR initializing EasyOCR: {e}")
                return None

        return EASYOCR_READER

    def recognize_with_easyocr(self, plate_image):
        """
        Recognize characters using EasyOCR

        Args:
            plate_image: Plate image (grayscale or BGR)

        Returns:
            Tuple of (text, boxes, confidences)
        """
        reader = self.get_easyocr_reader()
        if reader is None:
            return "EASYOCR_NOT_AVAILABLE", [], []

        try:
            # Convert to BGR if grayscale
            if len(plate_image.shape) == 2:
                plate_bgr = cv2.cvtColor(plate_image, cv2.COLOR_GRAY2BGR)
            else:
                plate_bgr = plate_image.copy()

            # Run EasyOCR
            results = reader.readtext(
                plate_bgr,
                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                detail=1,
                paragraph=False
            )

            if not results:
                return "NO_TEXT_DETECTED", [], []

            # Combine all detected text
            all_text = []
            all_boxes = []
            all_confidences = []

            for (bbox, text, confidence) in results:
                # Clean text
                cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
                if cleaned:
                    all_text.append(cleaned)
                    all_boxes.append(bbox)
                    all_confidences.append(confidence)

            # Join all text
            final_text = ''.join(all_text)
            avg_confidence = np.mean(all_confidences) if all_confidences else 0.0

            self.log_message(f"EasyOCR result: '{final_text}' (conf: {avg_confidence:.2%})")

            return final_text, all_boxes, all_confidences

        except Exception as e:
            self.log_message(f"ERROR in EasyOCR recognition: {e}")
            return "EASYOCR_ERROR", [], []

    def recognize_characters_ensemble(self, plate_image, binary_plate):
        """
        Ensemble voting between Tesseract and EasyOCR for best results

        Args:
            plate_image: Original plate image
            binary_plate: Binary thresholded plate image

        Returns:
            Tuple of (text, boxes, confidences, method_used)
        """
        results = []

        # Try Tesseract with segmentation
        if TESSERACT_AVAILABLE:
            try:
                text_seg, boxes_seg, conf_seg = self.recognize_characters_segmented(plate_image, binary_plate)
                if text_seg and text_seg not in ['NO_TEXT_DETECTED', 'SEGMENTATION_FAILED']:
                    avg_conf_seg = np.mean(conf_seg) if conf_seg else 0.0
                    results.append(('tesseract_segmented', text_seg, boxes_seg, conf_seg, avg_conf_seg))
                    self.log_message(f"Tesseract (segmented): '{text_seg}' (conf: {avg_conf_seg:.2%})")
            except Exception as e:
                self.log_message(f"Tesseract segmentation failed: {e}")

        # Try EasyOCR
        if EASYOCR_AVAILABLE:
            try:
                text_easy, boxes_easy, conf_easy = self.recognize_with_easyocr(plate_image)
                if text_easy and text_easy not in ['NO_TEXT_DETECTED', 'EASYOCR_NOT_AVAILABLE', 'EASYOCR_ERROR']:
                    avg_conf_easy = np.mean(conf_easy) if conf_easy else 0.0
                    results.append(('easyocr', text_easy, boxes_easy, conf_easy, avg_conf_easy))
            except Exception as e:
                self.log_message(f"EasyOCR failed: {e}")

        # If no results from either, fall back to standard Tesseract
        if not results and TESSERACT_AVAILABLE:
            text_std, boxes_std, conf_std = self.recognize_characters(plate_image, method='ocr')
            if text_std and text_std not in ['NO_TEXT_DETECTED', 'OCR_NOT_AVAILABLE']:
                avg_conf_std = np.mean(conf_std) if conf_std else 0.0
                results.append(('tesseract_standard', text_std, boxes_std, conf_std, avg_conf_std))
                self.log_message(f"Tesseract (fallback): '{text_std}' (conf: {avg_conf_std:.2%})")

        if not results:
            return "NO_TEXT_DETECTED", [], [], 'none'

        # Select best result based on:
        # 1. Confidence (primary)
        # 2. Text length matching Russian format (6-8 chars)
        # 3. Format validation
        best_result = None
        best_score = -1

        for method, text, boxes, confs, avg_conf in results:
            # Calculate score
            score = avg_conf  # Start with confidence

            # Bonus for correct length (Russian plates are typically 6-8 characters without spaces)
            cleaned_text = text.replace(' ', '')
            if 6 <= len(cleaned_text) <= 8:
                score += 0.1

            # Bonus for passing format validation
            if self.validate_plate_format(cleaned_text):
                score += 0.2

            self.log_message(f"Method '{method}': score={score:.3f} (conf={avg_conf:.2%}, len={len(cleaned_text)})")

            if score > best_score:
                best_score = score
                best_result = (method, text, boxes, confs)

        if best_result:
            method, text, boxes, confs = best_result
            self.log_message(f"✓ Selected: {method} with score {best_score:.3f}")
            return text, boxes, confs, method

        return "NO_TEXT_DETECTED", [], [], 'none'

    def recognize_characters(self, plate_image, method='ocr'):
        """
        Recognize characters on license plate

        Args:
            plate_image: Processed plate image
            method: 'cascade' or 'ocr'

        Returns:
            Recognized text, character bounding boxes, confidence scores
        """
        if method == 'ocr' and TESSERACT_AVAILABLE:
            # Use Tesseract OCR for character recognition
            if len(plate_image.shape) == 3:
                gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = plate_image

            # Invert if background is dark
            mean_val = np.mean(gray)
            if mean_val < 127:
                gray = cv2.bitwise_not(gray)

            # Try multiple PSM modes for best results
            psm_modes = [7, 8, 6, 13]  # single line, single word, uniform block, raw line
            best_text = ""
            best_confidence = 0.0
            best_boxes = []
            best_confidences = []

            for psm in psm_modes:
                # Configure Tesseract for license plate recognition
                custom_config = f'--oem 3 --psm {psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

                try:
                    # Extract text
                    text = pytesseract.image_to_string(gray, config=custom_config).strip()

                    # Clean text: remove spaces and special characters
                    text = re.sub(r'[^A-Z0-9]', '', text.upper())

                    if len(text) >= len(best_text):
                        # Get detailed data from Tesseract
                        data = pytesseract.image_to_data(gray, config=custom_config, output_type=pytesseract.Output.DICT)

                        # Get character bounding boxes
                        char_boxes = []
                        confidences = []
                        n_boxes = len(data['text'])
                        for i in range(n_boxes):
                            if int(data['conf'][i]) > 0:  # Filter low confidence
                                (x, y, w, h) = (data['left'][i], data['top'][i],
                                               data['width'][i], data['height'][i])
                                char_boxes.append((x, y, w, h))
                                confidences.append(float(data['conf'][i]) / 100.0)

                        avg_conf = np.mean(confidences) if confidences else 0.0

                        if text and (avg_conf > best_confidence or len(text) > len(best_text)):
                            best_text = text
                            best_confidence = avg_conf
                            best_boxes = char_boxes
                            best_confidences = confidences

                except Exception as e:
                    continue

            if best_text:
                return best_text, best_boxes, best_confidences
            else:
                return "NO_TEXT_DETECTED", [], []

        elif method == 'cascade' and self.char_cascade is not None:
            # Use cascade classifier for character detection
            if len(plate_image.shape) == 3:
                gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = plate_image

            chars = self.char_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=3, minSize=(5, 10)
            )

            if len(chars) > 0:
                # Sort characters left to right
                chars_sorted = sorted(chars, key=lambda x: x[0])
                return "CASCADE_DETECTED", chars_sorted, [1.0] * len(chars_sorted)
            else:
                return "NO_CHARS_DETECTED", [], []

        # Fallback: return placeholder
        return "OCR_NOT_AVAILABLE", [], []

    def correct_character_by_position(self, char, position):
        """
        Correct character based on its expected position in Russian plate
        Russian format: Letter(0) + Digit(1-3) + Letter(4-5)

        Args:
            char: Character to correct
            position: Position in plate (0-5)

        Returns:
            Corrected character
        """
        # Mapping of commonly confused characters
        LETTER_CORRECTIONS = {'0': 'O', '1': 'I', '8': 'B', '5': 'S', '6': 'G', '2': 'Z'}
        DIGIT_CORRECTIONS = {'O': '0', 'I': '1', 'B': '8', 'S': '5', 'G': '6', 'Z': '2',
                            'D': '0', 'Q': '0', 'L': '1', 'T': '7'}

        char = char.upper()

        # Positions 0, 4, 5 should be letters
        if position in [0, 4, 5]:
            if char in LETTER_CORRECTIONS:
                corrected = LETTER_CORRECTIONS[char]
                return corrected
            elif char.isdigit():
                # Digit in letter position, try to convert
                if char in LETTER_CORRECTIONS:
                    return LETTER_CORRECTIONS[char]
            return char

        # Positions 1, 2, 3 should be digits
        elif position in [1, 2, 3]:
            if char in DIGIT_CORRECTIONS:
                corrected = DIGIT_CORRECTIONS[char]
                return corrected
            elif char.isalpha():
                # Letter in digit position, try to convert
                if char in DIGIT_CORRECTIONS:
                    return DIGIT_CORRECTIONS[char]
            return char

        return char

    def validate_plate_format(self, text):
        """
        Validate and correct recognized text against Russian plate format
        Russian format: Letter + 3 digits + 2 letters (+ region code)
        Example: A123BC or A123BC77

        Args:
            text: Recognized text string

        Returns:
            is_valid: Boolean indicating if format matches
            formatted_text: Cleaned and formatted text with corrections
        """
        if not text or len(text) < 6:
            return False, text

        # Apply position-based corrections
        corrected = ""
        for i, char in enumerate(text[:6]):
            corrected += self.correct_character_by_position(char, i)

        # Extract region code if present
        region = text[6:] if len(text) > 6 else ""

        # Basic Russian plate pattern: Letter + 3 digits + 2 letters
        pattern = r'^[A-Z]\d{3}[A-Z]{2}$'

        if re.match(pattern, corrected):
            # Format nicely
            formatted = f"{corrected[0]} {corrected[1:4]} {corrected[4:6]}"
            if region:
                formatted += f" {region}"
            return True, formatted
        else:
            # Return corrected version even if doesn't match pattern
            formatted = f"{corrected}"
            if region:
                formatted += f" {region}"
            return False, formatted

    def draw_results(self, image, detections, text=None):
        """
        Visualize detection results on image

        Args:
            image: Input image
            detections: List of bounding boxes
            text: Recognized text (optional)

        Returns:
            Annotated image
        """
        result = image.copy()

        for i, (x, y, w, h) in enumerate(detections):
            # Draw red rectangle
            cv2.rectangle(result, (x, y), (x+w, y+h), (0, 0, 255), 3)

            # Add label
            label = f"Plate {i+1}"
            if text:
                label += f": {text}"

            # Add background for text
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(result, (x, y-text_h-10), (x+text_w, y), (0, 0, 255), -1)
            cv2.putText(result, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (255, 255, 255), 2)

        return result

    def create_results_grid(self, images_dict, titles):
        """
        Create a grid visualization of processing steps

        Args:
            images_dict: Dictionary of images
            titles: List of titles for each image

        Returns:
            Composite grid image
        """
        n = len(images_dict)
        if n == 0:
            return None

        # Create figure
        fig, axes = plt.subplots(1, n, figsize=(5*n, 5))
        if n == 1:
            axes = [axes]

        for ax, (key, img), title in zip(axes, images_dict.items(), titles):
            if len(img.shape) == 2:
                ax.imshow(img, cmap='gray')
            else:
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_title(title)
            ax.axis('off')

        plt.tight_layout()
        return fig

    def save_results(self, images, output_path, metadata):
        """
        Save processed images and metadata

        Args:
            images: Dictionary of images to save
            output_path: Base output path
            metadata: Dictionary with result metadata
        """
        for name, img in images.items():
            save_path = os.path.join(output_path, name)
            cv2.imwrite(save_path, img)
            self.log_message(f"Saved: {save_path}")

        # Save metadata as JSON
        json_path = os.path.join(output_path, 'metadata.json')
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        self.log_message(f"Saved metadata: {json_path}")

    def load_ground_truth(self, ground_truth_path):
        """
        Load ground truth plate numbers from JSON file

        Args:
            ground_truth_path: Path to ground_truth.json file

        Returns:
            Dictionary mapping image names to ground truth data
        """
        if not os.path.exists(ground_truth_path):
            self.log_message(f"WARNING: Ground truth file not found: {ground_truth_path}")
            return {}

        try:
            with open(ground_truth_path, 'r') as f:
                ground_truth = json.load(f)
            self.log_message(f"Loaded ground truth for {len(ground_truth)} images")
            return ground_truth
        except Exception as e:
            self.log_message(f"ERROR loading ground truth: {e}")
            return {}

    def calculate_character_accuracy(self, recognized, ground_truth):
        """
        Calculate character-level accuracy between recognized and ground truth text

        Args:
            recognized: Recognized plate text
            ground_truth: Ground truth plate text

        Returns:
            Accuracy percentage (0.0 to 1.0)
        """
        # Normalize: remove spaces and convert to uppercase
        recognized = recognized.replace(' ', '').upper()
        ground_truth = ground_truth.replace(' ', '').upper()

        if not ground_truth:
            return 0.0

        # Calculate character-by-character matches
        matches = 0
        max_len = max(len(recognized), len(ground_truth))

        for i in range(min(len(recognized), len(ground_truth))):
            if recognized[i] == ground_truth[i]:
                matches += 1

        # Accuracy = matches / max length (penalizes both insertions and deletions)
        accuracy = matches / max_len if max_len > 0 else 0.0
        return accuracy

    def validate_against_ground_truth(self, image_name, detected_plates, ground_truth_data):
        """
        Validate detected plates against ground truth

        Args:
            image_name: Name of the image being processed
            detected_plates: List of detected plate dictionaries
            ground_truth_data: Ground truth data for all images

        Returns:
            Dictionary with validation metrics
        """
        if image_name not in ground_truth_data:
            return {
                'has_ground_truth': False,
                'message': 'No ground truth available for this image'
            }

        gt_plates = ground_truth_data[image_name]['plates']

        results = {
            'has_ground_truth': True,
            'expected_count': len(gt_plates),
            'detected_count': len(detected_plates),
            'plate_accuracies': []
        }

        # Match detected plates to ground truth
        for gt_plate in gt_plates:
            gt_number = gt_plate['plate_number']
            best_match = None
            best_accuracy = 0.0

            for detected_plate in detected_plates:
                recognized = detected_plate.get('recognized_text', '')
                accuracy = self.calculate_character_accuracy(recognized, gt_number)

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_match = detected_plate

            plate_result = {
                'ground_truth': gt_number,
                'recognized': best_match.get('recognized_text', 'NO_TEXT_DETECTED') if best_match else 'NOT_DETECTED',
                'accuracy': best_accuracy,
                'position': gt_plate.get('position', 'unknown'),
                'notes': gt_plate.get('notes', '')
            }
            results['plate_accuracies'].append(plate_result)

        # Calculate overall accuracy for this image
        if results['plate_accuracies']:
            results['average_accuracy'] = sum(p['accuracy'] for p in results['plate_accuracies']) / len(results['plate_accuracies'])
        else:
            results['average_accuracy'] = 0.0

        return results

    def process_image(self, image_path, output_dir, preprocessing_level='medium',
                     detection_params=None):
        """
        Process a single image through the complete pipeline

        Args:
            image_path: Path to input image
            output_dir: Directory to save results
            preprocessing_level: Level of preprocessing to apply
            detection_params: Parameters for plate detection

        Returns:
            Dictionary with processing results
        """
        start_time = time.time()
        image_name = os.path.basename(image_path)
        self.log_message(f"\n{'='*60}")
        self.log_message(f"Processing: {image_name}")
        self.log_message(f"{'='*60}")

        # Load and preprocess
        original, gray, processed = self.load_and_preprocess(
            image_path, preprocessing_level
        )

        # Detect plates
        plates = self.detect_plates(original, processed, detection_params)

        result = {
            'image_name': image_name,
            'plates_detected': len(plates),
            'plates': [],
            'processing_time': 0
        }

        if len(plates) == 0:
            self.log_message("WARNING: No plates detected!")
            annotated = self.draw_results(original, [])

            # Save results
            output_images = {
                f'{image_name}_no_detection.jpg': annotated,
                f'{image_name}_processed.jpg': processed
            }

            img_output_dir = os.path.join(output_dir, 'annotated_originals')
            self.save_results(output_images, img_output_dir, result)

            result['processing_time'] = time.time() - start_time
            return result

        # Process each detected plate
        all_plate_data = []
        for i, (x, y, w, h) in enumerate(plates):
            self.log_message(f"\nProcessing plate {i+1}/{len(plates)}")

            plate_data = {
                'bbox': [int(x), int(y), int(w), int(h)],
                'aspect_ratio': float(w/h)
            }

            # Extract plate region
            plate_img = self.extract_plate_region(original, (x, y, w, h))

            # Align plate
            aligned_plate, rotation_angle = self.align_plate(plate_img)
            plate_data['rotation_angle'] = float(rotation_angle)

            # Process for OCR
            processed_plate = self.process_plate_for_ocr(aligned_plate)

            # Use ensemble method (combines Tesseract + EasyOCR)
            self.log_message("Running ensemble OCR (Tesseract + EasyOCR)...")
            text, char_boxes, confidences, ocr_method = self.recognize_characters_ensemble(
                aligned_plate, processed_plate
            )

            plate_data['recognized_text'] = text
            plate_data['ocr_method'] = ocr_method
            plate_data['num_characters'] = len(char_boxes)

            # Validate format
            is_valid, formatted_text = self.validate_plate_format(text)
            plate_data['format_valid'] = is_valid
            plate_data['formatted_text'] = formatted_text if is_valid else text

            if confidences:
                plate_data['avg_confidence'] = float(np.mean(confidences))
                plate_data['min_confidence'] = float(min(confidences))
                plate_data['max_confidence'] = float(max(confidences))

            self.log_message(f"Recognized: {text}")
            if is_valid:
                self.log_message(f"Formatted: {formatted_text}")

            # Save plate images
            plate_output_dir = os.path.join(output_dir, 'detected_plates')
            os.makedirs(plate_output_dir, exist_ok=True)
            cv2.imwrite(os.path.join(plate_output_dir, f'{image_name}_plate_{i+1}_raw.jpg'),
                       plate_img)
            cv2.imwrite(os.path.join(plate_output_dir, f'{image_name}_plate_{i+1}_aligned.jpg'),
                       aligned_plate)

            processed_output_dir = os.path.join(output_dir, 'processed_plates')
            os.makedirs(processed_output_dir, exist_ok=True)
            cv2.imwrite(os.path.join(processed_output_dir, f'{image_name}_plate_{i+1}_processed.jpg'),
                       processed_plate)

            all_plate_data.append(plate_data)

        result['plates'] = all_plate_data

        # Draw results on original image
        annotated = self.draw_results(original, plates)

        # Save annotated original
        annotated_output_dir = os.path.join(output_dir, 'annotated_originals')
        os.makedirs(annotated_output_dir, exist_ok=True)
        cv2.imwrite(os.path.join(annotated_output_dir, f'{image_name}_annotated.jpg'),
                   annotated)

        result['processing_time'] = time.time() - start_time
        self.log_message(f"Processing completed in {result['processing_time']:.2f}s")

        return result


def main():
    """Main execution function"""
    print("="*60)
    print("Russian License Plate Detection and Recognition")
    print("="*60)

    # Setup paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cascade_path = os.path.join(base_dir, 'cascades', 'haarcascade_russian_plate_number.xml')
    char_cascade_path = os.path.join(base_dir, 'cascades', 'haarcascade_license_plate_rus_16stages.xml')
    input_dir = os.path.join(base_dir, 'input_images')
    output_dir = os.path.join(base_dir, 'output')

    # Check cascade files exist
    if not os.path.exists(cascade_path):
        print(f"ERROR: Cascade file not found: {cascade_path}")
        return

    # Initialize detector
    detector = LicensePlateDetector(cascade_path, char_cascade_path)

    # Load ground truth data
    ground_truth_path = os.path.join(base_dir, 'ground_truth.json')
    ground_truth_data = detector.load_ground_truth(ground_truth_path)

    # Get all images from input directory
    image_files = [f for f in os.listdir(input_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]

    if len(image_files) == 0:
        print(f"ERROR: No images found in {input_dir}")
        return

    print(f"\nFound {len(image_files)} images to process")

    # Process each image with different parameter sets
    all_iterations = []

    # Iterations 6-7: Ground Truth + EasyOCR Ensemble
    print("\n" + "="*60)
    print("ITERATIONS 6-7: Ground Truth + EasyOCR Ensemble")
    print("="*60)
    print("Changes:")
    print("- Stricter detection parameters (minNeighbors: 3→5)")
    print("- Tighter aspect ratio validation (2.5-4.5)")
    print("- Color validation (white background check)")
    print("- Position heuristics (lower 2/3 of image)")
    print("- Adaptive upscaling (3-6x based on plate size)")
    print("- Sharpening filter + better denoising")
    print("- Position-based character correction")
    print("- ENSEMBLE OCR (New!)")
    print("  * Combines Tesseract + EasyOCR")
    print("  * Intelligent voting system")
    print("  * Selects best result based on confidence + format validation")
    print("- GROUND TRUTH VALIDATION (New!)")
    print("  * Manual plate transcriptions for accuracy measurement")
    print("  * Character-level accuracy calculation")
    print("  * Perfect match detection")
    print("="*60)

    all_results = []
    detector.log = []  # Reset log

    for img_file in image_files:
        img_path = os.path.join(input_dir, img_file)
        result = detector.process_image(
            img_path,
            output_dir,
            preprocessing_level='medium',
            detection_params={
                'scaleFactor': 1.05,    # Smaller steps for more thorough detection
                'minNeighbors': 5,      # Stricter filtering (was 3)
                'minSize': (40, 12)     # Slightly larger minimum (was 30, 10)
            }
        )

        # Validate against ground truth if available
        if ground_truth_data:
            validation = detector.validate_against_ground_truth(
                img_file, result.get('plates', []), ground_truth_data
            )
            result['ground_truth_validation'] = validation

        all_results.append(result)

    # Save iteration 2-3 results
    results_json_path = os.path.join(output_dir, 'results_iteration_2-3.json')
    with open(results_json_path, 'w') as f:
        json.dump({
            'iteration': '2-3',
            'description': 'Improved detection filtering + enhanced OCR preprocessing',
            'changes': {
                'detection': {
                    'scaleFactor': 1.05,
                    'minNeighbors': 5,
                    'minSize': [40, 12],
                    'aspect_ratio_range': [2.5, 4.5],
                    'color_validation': True,
                    'position_heuristics': True
                },
                'ocr_preprocessing': {
                    'adaptive_upscaling': '3-6x based on size',
                    'denoising': 'fastNlMeans',
                    'sharpening': True,
                    'multiple_thresholding': True
                },
                'character_correction': {
                    'position_based': True,
                    'corrections': 'O<->0, I<->1, B<->8, etc.'
                }
            },
            'results': all_results,
            'log': detector.log
        }, f, indent=2)

    all_iterations.append({
        'iteration': '2-3',
        'results': all_results
    })

    print("\n" + "="*60)
    print("SUMMARY - ITERATION 2 & 3")
    print("="*60)
    print(f"Total images processed: {len(all_results)}")
    print(f"Images with detections: {sum(1 for r in all_results if r['plates_detected'] > 0)}")
    print(f"Total plates detected: {sum(r['plates_detected'] for r in all_results)}")

    # Detailed breakdown
    print("\nDetailed Results:")
    for result in all_results:
        print(f"\n{result['image_name']}:")
        print(f"  Plates detected: {result['plates_detected']}")
        for i, plate in enumerate(result.get('plates', [])):
            text = plate.get('recognized_text', 'N/A')
            formatted = plate.get('formatted_text', text)
            valid = plate.get('format_valid', False)
            conf = plate.get('avg_confidence', 0.0)
            method = plate.get('ocr_method', 'unknown')
            print(f"  Plate {i+1}: {text}")
            print(f"    Method: {method}")
            if valid:
                print(f"    Corrected: {formatted} ✓")
            else:
                print(f"    Attempted correction: {formatted}")
            if conf > 0:
                print(f"    Confidence: {conf:.2%}")
                min_conf = plate.get('min_confidence', 0)
                max_conf = plate.get('max_confidence', 0)
                if min_conf > 0 and max_conf > 0:
                    print(f"    Range: {min_conf:.2%} - {max_conf:.2%}")

    # Count format validation successes
    valid_plates = sum(1 for r in all_results for p in r.get('plates', []) if p.get('format_valid', False))
    total_plates = sum(r['plates_detected'] for r in all_results)

    print(f"\nFormat Validation:")
    print(f"  Valid Russian format: {valid_plates}/{total_plates}")

    # Ground Truth Accuracy (if available)
    if ground_truth_data:
        print(f"\n{'='*60}")
        print("GROUND TRUTH ACCURACY")
        print(f"{'='*60}")

        all_accuracies = []
        for result in all_results:
            validation = result.get('ground_truth_validation', {})
            if validation.get('has_ground_truth', False):
                print(f"\n{result['image_name']}:")
                print(f"  Expected plates: {validation['expected_count']}")
                print(f"  Detected plates: {validation['detected_count']}")

                for plate_acc in validation.get('plate_accuracies', []):
                    gt = plate_acc['ground_truth']
                    rec = plate_acc['recognized']
                    acc = plate_acc['accuracy']
                    all_accuracies.append(acc)

                    print(f"  Ground Truth: {gt}")
                    print(f"  Recognized:   {rec}")
                    print(f"  Accuracy:     {acc:.1%}")
                    if acc == 1.0:
                        print(f"    ✓ PERFECT MATCH!")
                    elif acc >= 0.8:
                        print(f"    ✓ Good match")
                    elif acc >= 0.5:
                        print(f"    ~ Partial match")
                    else:
                        print(f"    ✗ Poor match")

        if all_accuracies:
            avg_accuracy = sum(all_accuracies) / len(all_accuracies)
            perfect_matches = sum(1 for a in all_accuracies if a == 1.0)
            good_matches = sum(1 for a in all_accuracies if a >= 0.8)

            print(f"\n{'='*60}")
            print("OVERALL ACCURACY METRICS")
            print(f"{'='*60}")
            print(f"  Average Character Accuracy: {avg_accuracy:.1%}")
            print(f"  Perfect Matches (100%): {perfect_matches}/{len(all_accuracies)}")
            print(f"  Good Matches (≥80%): {good_matches}/{len(all_accuracies)}")

    print(f"\nResults saved to: {output_dir}")
    print(f"Detailed log saved to: {results_json_path}")
    print("\nCheck output/annotated_originals/ for visual results")


if __name__ == "__main__":
    main()
