"""Core OCR utilities for the Batch OCR application."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import cv2  # type: ignore
import numpy as np
import pytesseract
from langdetect import LangDetectException, detect
from PIL import Image, UnidentifiedImageError

LOGGER = logging.getLogger("batch_ocr")


def init_logging(log_path: Path = Path("ocr_app.log")) -> None:
    """Initialize application-wide logging."""
    if LOGGER.handlers:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)


def find_tesseract_executable() -> Optional[str]:
    """Try to locate Tesseract executable in common Windows locations."""
    import platform
    import shutil
    
    # First, check if tesseract is in PATH
    tesseract_path = shutil.which("tesseract")
    if tesseract_path:
        return tesseract_path
    
    # Check common Windows installation paths
    if platform.system() == "Windows":
        common_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            r"C:\Tesseract-OCR\tesseract.exe",
        ]
        for path in common_paths:
            if Path(path).exists():
                return path
    
    return None


def configure_tesseract(custom_path: Optional[str] = None) -> Optional[str]:
    """Configure pytesseract with Tesseract executable path. Returns path if found, None otherwise."""
    if custom_path and Path(custom_path).exists():
        pytesseract.pytesseract.tesseract_cmd = custom_path
        return custom_path
    
    # Try to auto-detect
    detected_path = find_tesseract_executable()
    if detected_path:
        pytesseract.pytesseract.tesseract_cmd = detected_path
        return detected_path
    
    return None


@dataclass
class OCRResult:
    """Result payload for a single OCR run."""

    source_path: Path
    output_path: Optional[Path]
    text: str
    success: bool
    error: Optional[str] = None


PREPROCESSING_PRESETS: Dict[str, Dict[str, int]] = {
    "printed": {
        "target_width": 1500,
        "blur": 3,  # median blur strength
        "use_bilateral": 0,
        "clahe_clip": 2,
        "adaptive_block": 21,
        "adaptive_c": 9,
        "morph_kernel": 2,
        "apply_fast_denoise": 0,
    },
    "handwritten": {
        "target_width": 1800,
        "blur": 5,
        "use_bilateral": 1,
        "clahe_clip": 3,
        "adaptive_block": 25,
        "adaptive_c": 11,
        "morph_kernel": 1,
        "apply_fast_denoise": 1,
    },
}


def preprocess_image(
    image_path: Union[Path, str], preset: str = "handwritten"
) -> Tuple[np.ndarray, np.ndarray]:
    """Load and preprocess an image to improve OCR accuracy."""

    params = PREPROCESSING_PRESETS.get(preset, PREPROCESSING_PRESETS["handwritten"])
    image_path = Path(image_path)
    bgr_image = cv2.imread(str(image_path))
    if bgr_image is None:
        try:
            with Image.open(image_path) as pil_img:
                bgr_image = cv2.cvtColor(
                    np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR
                )
        except (FileNotFoundError, UnidentifiedImageError) as exc:
            raise ValueError(f"Unsupported or unreadable image: {image_path}") from exc

    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    target_width = params["target_width"]
    if width < target_width:
        scale = target_width / float(width)
        new_size = (int(width * scale), int(height * scale))
        gray = cv2.resize(gray, new_size, interpolation=cv2.INTER_CUBIC)

    if params["use_bilateral"]:
        gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    else:
        blur_strength = max(3, params["blur"] | 1)  # ensure odd kernel
        gray = cv2.medianBlur(gray, blur_strength)

    clahe = cv2.createCLAHE(clipLimit=params["clahe_clip"], tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    block_size = params["adaptive_block"]
    if block_size % 2 == 0:
        block_size += 1
    threshold = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        params["adaptive_c"],
    )

    kernel_size = max(1, params["morph_kernel"])
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    cleaned = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)

    if params["apply_fast_denoise"]:
        cleaned = cv2.fastNlMeansDenoising(cleaned, h=30)

    return bgr_image, cleaned


def _detect_language(text: str) -> Optional[str]:
    """Use langdetect to infer the best Tesseract language code."""
    stripped = text.strip()
    if not stripped:
        return None
    try:
        detected = detect(stripped)
    except LangDetectException:
        return None
    if detected.startswith("hi"):
        return "hin"
    if detected.startswith("en"):
        return "eng"
    return None


def _build_config(oem: int, psm: int) -> str:
    """Compose the pytesseract configuration string."""
    return f"--oem {oem} --psm {psm}"


def save_text_output(
    text: str,
    image_path: Union[Path, str],
    output_dir: Optional[Path],
    save_to_source: bool = True,
    overwrite: bool = False,
) -> Path:
    """Persist OCR text to a UTF-8 file respecting overwrite rules."""
    image_path = Path(image_path)
    target_dir = image_path.parent if save_to_source or not output_dir else output_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    base_name = f"output_{stem}.txt"
    output_path = target_dir / base_name
    if output_path.exists() and not overwrite:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = target_dir / f"output_{stem}_{timestamp}.txt"
    output_path.write_text(text, encoding="utf-8", errors="replace")
    return output_path


def perform_ocr(
    processed_image: np.ndarray,
    lang: str,
    oem: int = 3,
    psm: int = 6,
) -> str:
    """Run pytesseract on the processed image."""
    config = _build_config(oem=oem, psm=psm)
    return pytesseract.image_to_string(processed_image, lang=lang, config=config)


def process_image(
    image_path: Union[Path, str],
    lang_choice: str = "eng+hin",
    preset: str = "handwritten",
    save_to_source: bool = True,
    output_dir: Optional[Path] = None,
    overwrite: bool = False,
    enable_auto_lang: bool = False,
    oem: int = 3,
    psm: int = 6,
) -> OCRResult:
    """Run preprocessing, OCR, and save the result for a single file."""
    image_path = Path(image_path)
    try:
        _original, processed = preprocess_image(image_path, preset=preset)
        lang_to_use = "eng+hin" if lang_choice in {"auto", "eng+hin"} else lang_choice
        text = perform_ocr(processed, lang=lang_to_use, oem=oem, psm=psm)

        if enable_auto_lang and lang_choice == "auto":
            detected = _detect_language(text)
            if detected and detected != "eng+hin":
                LOGGER.info("Detected %s for %s, re-running OCR", detected, image_path)
                text = perform_ocr(processed, lang=detected, oem=oem, psm=psm)

        output_path = save_text_output(
            text,
            image_path=image_path,
            output_dir=output_dir,
            save_to_source=save_to_source,
            overwrite=overwrite,
        )
        LOGGER.info("OCR success: %s -> %s", image_path.name, output_path.name)
        return OCRResult(source_path=image_path, output_path=output_path, text=text, success=True)
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.exception("Failed to process %s", image_path)
        return OCRResult(
            source_path=image_path,
            output_path=None,
            text="",
            success=False,
            error=str(exc),
        )

