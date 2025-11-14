# Batch OCR — Image to .txt

Tkinter desktop application for converting batches of printed or handwritten English/Hindi images into UTF-8 `.txt` files using Tesseract OCR.

## Features

- Multiple image selection with list view and removal controls
- Preprocessing presets for printed vs. handwritten sources (OpenCV + Pillow)
- Supports English, Hindi, or combined `eng+hin` recognition with optional language detection
- Progress bar, live status, and in-app preview of OCR results
- Auto-saving `output_<filename>.txt` next to each image or in a custom folder
- Robust logging (`ocr_app.log`) and per-file error isolation
- Threaded processing to keep the GUI responsive

## Requirements

- Windows 11 (tested) or any OS with Python 3.9+
- Python 3.9, 3.10, 3.11, or 3.12
- Tesseract OCR engine (UB Mannheim Windows build recommended)
- Pip packages listed in `requirements.txt`

## Setup

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Install Tesseract on Windows 11

1. Download the UB Mannheim build: https://github.com/UB-Mannheim/tesseract/wiki
2. Run the installer and note the destination, typically `C:\Program Files\Tesseract-OCR\tesseract.exe`
3. Ensure `hin.traineddata` exists inside `Tesseract-OCR\tessdata`; add it if missing
4. Add the install directory to the PATH environment variable **or** update `main.py` to point `pytesseract.pytesseract.tesseract_cmd` at the executable

## Running the App

```powershell
python main.py
```

Expected flow:

1. Click **Select Images** and choose `.jpg/.jpeg/.png/.bmp` files
2. Verify the file list, adjust language/preset/OEM/PSM as needed
3. Choose whether to save outputs next to the images or in another folder
4. Press **Start OCR** to process all files; view text output live and review the final summary dialog

## Preprocessing Pipeline

Each image runs through the following configurable steps (see `ocr_core.py` for exact parameters):

1. Load via OpenCV with Pillow fallback
2. Convert to grayscale
3. Upscale to ~1500–1800 px width using cubic interpolation if the input is narrow
4. Apply median blur (printed) or bilateral filter (handwritten) for denoising
5. Enhance contrast using CLAHE to highlight faint handwriting
6. Adaptive thresholding (Gaussian) with tunable block size and `C`
7. Morphological open/close (1–2 px kernels) to remove speckles
8. Optional `fastNlMeansDenoising` for noisy handwritten scans

## Tests

```powershell
pytest tests/
```

`tests/test_file_output.py` mocks Tesseract to verify that processing a sample image generates a UTF-8 text file.

## Troubleshooting

- **Tesseract not found**: Confirm the executable path and update `main.py` or add the folder to PATH
- **Missing `hin.traineddata`**: Copy it into `Tesseract-OCR/tessdata`
- **Encoding errors**: Files are saved with `encoding="utf-8"` and `errors="replace"`; ensure downstream tools read UTF-8
- **Unsupported files**: Only `.jpg/.jpeg/.png/.bmp` are processed; others are skipped with warnings

## Assets

`assets/sample_images/README.md` documents placeholder filenames—add your own images there for testing.

## Git Tips

- Initial commit: `feat: initial Tkinter batch OCR app (printed + handwritten, eng+hin)`
- Example follow-ups: `fix: handle missing tesseract.exe`, `feat: add preprocessing presets`