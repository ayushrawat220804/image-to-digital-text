"""Tkinter GUI for the Batch OCR application."""

from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from typing import List, Optional

import pytesseract
from ocr_core import OCRResult, configure_tesseract, find_tesseract_executable, init_logging, process_image

WINDOW_TITLE = "Batch OCR — Image to .txt"
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
DEFAULT_TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


class BatchOCRApp:
    """Encapsulates the Tkinter GUI and event handling."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.geometry("900x700")

        self.selected_files: List[Path] = []
        self.is_processing = False

        self.language_var = tk.StringVar(value="eng+hin")
        self.save_same_var = tk.BooleanVar(value=True)
        self.overwrite_var = tk.BooleanVar(value=False)
        self.langdetect_var = tk.BooleanVar(value=False)
        self.output_dir: Optional[Path] = None
        self.preset_var = tk.StringVar(value="handwritten")
        self.psm_var = tk.IntVar(value=6)
        self.oem_var = tk.IntVar(value=3)

        self._build_ui()

    def _build_ui(self) -> None:
        controls_frame = ttk.Frame(self.root, padding=10)
        controls_frame.pack(fill=tk.X)

        select_btn = ttk.Button(controls_frame, text="Select Images", command=self.select_images)
        select_btn.grid(row=0, column=0, padx=(0, 5), pady=5, sticky="w")

        remove_btn = ttk.Button(controls_frame, text="Remove Selected", command=self.remove_selected)
        remove_btn.grid(row=0, column=1, padx=(0, 5), pady=5, sticky="w")

        ttk.Label(controls_frame, text="Language:").grid(row=1, column=0, sticky="w")
        lang_combo = ttk.Combobox(
            controls_frame,
            textvariable=self.language_var,
            values=("auto", "eng", "hin", "eng+hin"),
            state="readonly",
            width=15,
        )
        lang_combo.grid(row=1, column=1, sticky="w", pady=5)

        ttk.Label(controls_frame, text="Preset:").grid(row=1, column=2, sticky="w")
        preset_combo = ttk.Combobox(
            controls_frame,
            textvariable=self.preset_var,
            values=("printed", "handwritten"),
            state="readonly",
            width=15,
        )
        preset_combo.grid(row=1, column=3, sticky="w", pady=5)

        ttk.Label(controls_frame, text="OEM:").grid(row=2, column=0, sticky="w")
        oem_spin = ttk.Spinbox(controls_frame, from_=0, to=3, textvariable=self.oem_var, width=5)
        oem_spin.grid(row=2, column=1, sticky="w")

        ttk.Label(controls_frame, text="PSM:").grid(row=2, column=2, sticky="w")
        psm_spin = ttk.Spinbox(controls_frame, from_=0, to=13, textvariable=self.psm_var, width=5)
        psm_spin.grid(row=2, column=3, sticky="w")

        save_check = ttk.Checkbutton(
            controls_frame,
            text="Save .txt next to image",
            variable=self.save_same_var,
            command=self._toggle_output_controls,
        )
        save_check.grid(row=3, column=0, columnspan=2, sticky="w", pady=5)

        overwrite_check = ttk.Checkbutton(
            controls_frame,
            text="Overwrite existing output files",
            variable=self.overwrite_var,
        )
        overwrite_check.grid(row=3, column=2, columnspan=2, sticky="w", pady=5)

        langdetect_check = ttk.Checkbutton(
            controls_frame,
            text="Auto-detect language (experimental)",
            variable=self.langdetect_var,
        )
        langdetect_check.grid(row=4, column=0, columnspan=4, sticky="w")

        self.output_dir_btn = ttk.Button(
            controls_frame,
            text="Choose output folder…",
            command=self.choose_output_folder,
            state=tk.DISABLED,
        )
        self.output_dir_btn.grid(row=5, column=0, columnspan=2, sticky="w", pady=5)

        list_frame = ttk.LabelFrame(self.root, text="Selected images", padding=10)
        list_frame.pack(fill=tk.BOTH, expand=False, padx=10, pady=5)

        self.listbox = tk.Listbox(list_frame, height=8, selectmode=tk.EXTENDED)
        self.listbox.pack(fill=tk.BOTH, expand=True)

        progress_frame = ttk.Frame(self.root, padding=10)
        progress_frame.pack(fill=tk.X)
        self.progress = ttk.Progressbar(progress_frame, mode="determinate")
        self.progress.pack(fill=tk.X)
        self.status_var = tk.StringVar(value="Idle")
        status_label = ttk.Label(progress_frame, textvariable=self.status_var)
        status_label.pack(fill=tk.X, pady=(5, 0))

        # Start OCR button - placed prominently before the output area
        button_frame = ttk.Frame(self.root, padding=10)
        button_frame.pack(fill=tk.X, pady=10)
        self.start_btn = ttk.Button(
            button_frame, text="▶ Start OCR", command=self.start_ocr, width=20
        )
        self.start_btn.pack()

        text_frame = ttk.LabelFrame(self.root, text="OCR output", padding=10)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self.output_text = ScrolledText(text_frame, wrap=tk.WORD, state=tk.DISABLED)
        self.output_text.pack(fill=tk.BOTH, expand=True)

        self._refresh_output_button_state()

    def select_images(self) -> None:
        file_paths = filedialog.askopenfilenames(
            title="Select image files",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")],
        )
        for path_str in file_paths:
            path = Path(path_str)
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                messagebox.showwarning("Unsupported file", f"{path.name} is not a supported image.")
                continue
            if path not in self.selected_files:
                self.selected_files.append(path)
                self.listbox.insert(tk.END, str(path))

    def remove_selected(self) -> None:
        selections = list(self.listbox.curselection())
        for index in reversed(selections):
            self.listbox.delete(index)
            del self.selected_files[index]

    def choose_output_folder(self) -> None:
        directory = filedialog.askdirectory(title="Select output folder")
        if directory:
            self.output_dir = Path(directory)
            self.status_var.set(f"Output folder: {self.output_dir}")

    def _toggle_output_controls(self) -> None:
        self._refresh_output_button_state()

    def _refresh_output_button_state(self) -> None:
        if self.is_processing:
            self.output_dir_btn.configure(state=tk.DISABLED)
            return
        if self.save_same_var.get():
            self.output_dir_btn.configure(state=tk.DISABLED)
        else:
            self.output_dir_btn.configure(state=tk.NORMAL)

    def start_ocr(self) -> None:
        if self.is_processing:
            return
        if not self.selected_files:
            messagebox.showinfo("No files", "Please select at least one image.")
            return
        
        # Check if Tesseract is available
        tesseract_cmd = getattr(pytesseract.pytesseract, 'tesseract_cmd', None)
        if not tesseract_cmd or not Path(tesseract_cmd).exists():
            tesseract_path = find_tesseract_executable()
            if not tesseract_path:
                error_msg = (
                    "Tesseract OCR is not installed or not found.\n\n"
                    "Please install Tesseract OCR:\n"
                    "1. Download from: https://github.com/UB-Mannheim/tesseract/wiki\n"
                    "2. Install to default location (C:\\Program Files\\Tesseract-OCR)\n"
                    "3. Restart this application\n\n"
                    "Or add Tesseract to your system PATH."
                )
                messagebox.showerror("Tesseract Not Found", error_msg)
                return
            else:
                pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
        self.is_processing = True
        self.start_btn.configure(state=tk.DISABLED)
        self._refresh_output_button_state()
        self.progress.configure(maximum=len(self.selected_files), value=0)
        self._set_status("Starting OCR…")
        self._set_text("")
        thread = threading.Thread(target=self._run_batch, daemon=True)
        thread.start()

    def _run_batch(self) -> None:
        total = len(self.selected_files)
        successes = 0
        failures: List[str] = []
        for idx, image_path in enumerate(self.selected_files, start=1):
            lang_choice = self.language_var.get()
            result = process_image(
                image_path=image_path,
                lang_choice=lang_choice,
                preset=self.preset_var.get(),
                save_to_source=self.save_same_var.get(),
                output_dir=self.output_dir,
                overwrite=self.overwrite_var.get(),
                enable_auto_lang=self.langdetect_var.get(),
                oem=self.oem_var.get(),
                psm=self.psm_var.get(),
            )
            if result.success and result.output_path:
                successes += 1
                self._display_result(result)
            else:
                failures.append(f"{image_path.name}: {result.error}")
            self._update_progress(idx, total, image_path.name)
        self.is_processing = False
        summary = f"Processed {total} file(s).\nSuccess: {successes}\nFailed: {len(failures)}"
        if failures:
            summary += "\n\nIssues:\n" + "\n".join(failures[:5])
        self.root.after(0, lambda: self._finalize_batch(summary))

    def _finalize_batch(self, summary: str) -> None:
        self.is_processing = False
        self.start_btn.configure(state=tk.NORMAL)
        self._refresh_output_button_state()
        messagebox.showinfo("Batch complete", summary)

    def _display_result(self, result: OCRResult) -> None:
        header = f"===== {result.source_path.name} =====\n"
        body = f"{result.text.strip()}\n\nSaved to: {result.output_path}\n\n"
        self.root.after(0, lambda: self._append_text(header + body))

    def _update_progress(self, current: int, total: int, filename: str) -> None:
        self.root.after(
            0,
            lambda: (
                self.progress.configure(value=current),
                self._set_status(f"Processing {current}/{total} — {filename}"),
            ),
        )

    def _set_status(self, text: str) -> None:
        self.status_var.set(text)

    def _append_text(self, text: str) -> None:
        self.output_text.configure(state=tk.NORMAL)
        self.output_text.insert(tk.END, text + "\n")
        self.output_text.configure(state=tk.DISABLED)
        self.output_text.see(tk.END)

    def _set_text(self, text: str) -> None:
        self.output_text.configure(state=tk.NORMAL)
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, text)
        self.output_text.configure(state=tk.DISABLED)


def main() -> None:
    """Launch the Tkinter event loop."""
    init_logging()
    # Try to configure Tesseract - check default path first, then auto-detect
    default_binary = Path(DEFAULT_TESSERACT_PATH)
    tesseract_path = configure_tesseract(
        custom_path=str(default_binary) if default_binary.exists() else None
    )
    
    if not tesseract_path:
        # Show warning but allow app to start - user can still try to use it
        print("Warning: Tesseract not found. OCR will fail until Tesseract is installed.")
        print("Download from: https://github.com/UB-Mannheim/tesseract/wiki")
    
    root = tk.Tk()
    app = BatchOCRApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

