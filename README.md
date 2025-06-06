# seg-background

`seg-background` is a simple command-line tool to segment background from organ image stacks using the Segment Anything Model 2 (SAM2). Given a folder of 2D TIFF or JP2 slices, it:

1. Applies median filtering and intensity normalization.
2. Builds forward/backward videos for SAM2 propagation.
3. Runs SAM2 to generate a binary mask for each slice.
4. Saves masks under an output directory (`masks/mask_0000.png`, etc.).

> **GPU Required:** SAM2 needs a CUDA-enabled NVIDIA GPU (CUDA ≥ 11.1).

---

## Installation

1. **Clone and enter the repo**  
   ```bash
   git clone https://github.com/JosephBrunet/seg-background.git
   cd seg-background
   ```

2. **Create a Python 3.10+ virtual environment**  
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install**  
   ```bash
   pip install --upgrade pip
   pip install .
   ```

4. **Verify**  
   ```bash
   seg-background --help
   ```
   You should see:
   ```
   usage: seg-background [-h] [--output OUTPUT] im_dir
   ```

---

## Usage

```bash
seg-background /path/to/image_folder --output /path/to/results
```

- **`/path/to/image_folder`**  
  Folder containing `.tif` or `.jp2` slices (sorted alphanumerically).

- **`--output /path/to/results`** _(optional)_  
  Base directory for outputs. Inside you’ll find:
  ```
  /path/to/results/segmentation_SAM2/
      ├── video_forward.mp4
      ├── video_backward.mp4
      └── masks/
          ├── mask_0000.png
          ├── mask_0001.png
          └── …
  ```

A window showing the middle slice will open. Click and drag the blue rectangle to create a rectangle covering the organ:

<p align="center">
  <img src="assets/images/sam2_window.png" alt="SAM2 ROI Selection Window" width="50%" />
</p>

The script will then convert to video and segment the whole volume.

<p align="center">
  <img src="assets/images/sam2_result.png" alt="SAM2 ROI Selection Window" width="50%" />
</p>

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
