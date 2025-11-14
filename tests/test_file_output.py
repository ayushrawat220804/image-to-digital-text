"""Unit test for verifying OCR output file creation."""

from pathlib import Path

from PIL import Image, ImageDraw

from ocr_core import OCRResult, process_image


def _create_sample_image(tmp_path: Path) -> Path:
    image_path = tmp_path / "sample_text.png"
    image = Image.new("RGB", (400, 200), color="white")
    draw = ImageDraw.Draw(image)
    draw.text((10, 80), "Hello OCR", fill="black")
    image.save(image_path)
    return image_path


def test_output_file_creation(monkeypatch, tmp_path) -> None:
    """Ensure that processing an image writes a UTF-8 text file."""
    sample_image = _create_sample_image(tmp_path)

    def fake_image_to_string(*_, **__) -> str:
        return "dummy text from test"

    monkeypatch.setattr("ocr_core.pytesseract.image_to_string", fake_image_to_string)

    result: OCRResult = process_image(
        image_path=sample_image,
        lang_choice="eng",
        preset="printed",
        save_to_source=False,
        output_dir=tmp_path,
        overwrite=True,
        enable_auto_lang=False,
    )

    assert result.success
    assert result.output_path is not None
    output_text = result.output_path.read_text(encoding="utf-8")
    assert "dummy text from test" in output_text

