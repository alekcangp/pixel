# Pixel

**Local-first image deduplication, clustering, and semantic search for your entire computer.**

[English](README.en.md) · [Русский](README.md)

> Organize thousands of images without manually browsing folders.
> Scan your drives, find visually similar images, group them into clusters, search by natural language, select the images you need, and export them — all locally.

![Image Deduplication UI](screenshots/main.png)

## Overview

**Image Deduplication** is a local desktop application and CLI for organizing large image collections.

Instead of opening folders one by one, the application builds a unified local index of your images and processes them through several stages:

```text
Scan
  ↓
Perceptual deduplication
  ↓
Image embeddings
  ↓
Semantic clustering
  ↓
Visual selection
  ↓
Export
```

The application stores its index and metadata locally and does not require uploading your images to a cloud service.

## Features

* 🔎 **Incremental scanning** — only new or changed files are reprocessed.
* 🧹 **Perceptual deduplication** — finds visually similar images using pHash + LSH.
* 🧠 **Semantic search** — search images using natural-language queries with SigLIP and FAISS.
* 🗂️ **Automatic clustering** — groups visually/semantically related images using UMAP + HDBSCAN.
* 🖥️ **Desktop UI** — interactive gallery built with Flet.
* 🖱️ **Visual selection** — select individual images or entire groups.
* 📦 **Export** — copy selected images to a destination folder.
* 💾 **Local processing** — image data and the local index remain on your machine.
* ⚡ **GPU acceleration** — CUDA is used on supported NVIDIA systems; MPS is used on Apple Silicon; otherwise CPU is used.

## How it works

### 1. Scan

The scanner recursively walks the selected directories and indexes supported image files.

Supported extensions by default:

* JPEG / JPG
* PNG
* WebP
* BMP
* GIF
* TIFF

Scanning is incremental and tracks file metadata/content information so unchanged files do not need to be processed again.

The GUI can scan all available disks, while system, cache, development, and other service directories are excluded by default.

### 2. Deduplication

Each image is converted into a perceptual hash (pHash).

A Locality-Sensitive Hashing (LSH) index is then used to efficiently find hashes within a configurable Hamming-distance threshold.

The current default threshold is:

```text
pHash Hamming distance <= 6
```

Similar hashes are grouped using Union-Find.

The deduplication stage can optionally move all but one file from each similar group to another directory.

> **Important:** pHash detects perceptual similarity. It is not an exact byte-for-byte file comparison.

### 3. Image embeddings

For semantic search and clustering, images are converted into embeddings using:

```text
google/siglip-base-patch16-224
```

The embeddings are stored locally and can be reused between runs.

The application automatically selects the best available PyTorch device:

```text
NVIDIA GPU → CUDA
Apple Silicon → MPS
otherwise   → CPU
```

### 4. Clustering

Image embeddings are reduced with UMAP and clustered with HDBSCAN.

The implementation also contains additional processing for large clusters and noisy points.

Clustering is automatically performed by the desktop workflow after embeddings are generated.

### 5. Search

Semantic search accepts natural-language queries such as:

```text
cat on a window
mountains at sunset
people on the beach
red car
```

The text query is embedded and compared with image embeddings using a FAISS inner-product index.

### 6. Select and export

The desktop UI provides a gallery for reviewing images.

You can:

* select individual images;
* select all images in a group;
* inspect images in a larger preview;
* search the indexed collection;
* review selected files;
* export selected files to a directory.

## Requirements

* Python **3.12** recommended
* pip
* Internet connection for the initial model download

For semantic search, embeddings, and the desktop UI, PyTorch and the Hugging Face model are required.

## Installation

For a quick setup, just download and run the installer file for your operating system. The installer will create the Python 3.12 virtual environment, install dependencies, download the model, and start the application automatically.

### macOS

Run:

```bash
chmod +x install_and_run.command
./install_and_run.command
```

The installer creates a Python 3.12 virtual environment, installs dependencies, downloads the SigLIP model, and starts the Flet desktop application.

> The bundled `.command` installer is currently intended for macOS. For Linux, use the manual installation procedure below.

### Windows

Run:

```bat
install_and_run.bat
```

The Windows installer downloads/configures Python 3.12 when necessary, installs the required runtime components and Python dependencies, downloads the model, and starts the application.

### Manual installation

Clone the repository:

```bash
git clone https://github.com/alekcangp/pixel.git
cd pixel
```

Create a virtual environment.

**Windows — Command Prompt:**

```bat
python -m venv .venv
.venv\Scripts\activate.bat
```

**Windows — PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Download the image-text model:

```bash
python -c "from transformers import AutoProcessor, AutoModel; m='google/siglip-base-patch16-224'; AutoProcessor.from_pretrained(m); AutoModel.from_pretrained(m); print('Model loaded:', m)"
```

## Quick Start

### Desktop application

```bash
python main.py ui-flet
```

The desktop UI provides scanning, deduplication, clustering, semantic search, image selection, and export in one workflow.

### CLI

#### Scan a directory

```bash
python main.py scan --path "/path/to/folder"
```

#### Force a full rescan

```bash
python main.py scan --path "/path/to/folder" --full
```

#### Scan with extensions and exclusions

```bash
python main.py scan \
  --path "/path/to/folder" \
  --ext "jpg,png,webp" \
  --exclude "tmp,cache"
```

#### Change the minimum file size

The default minimum file size is **20 KiB**.

```bash
python main.py scan --path "/path/to/folder" --min-size 5
```

The value is specified in KiB.

#### Scan and deduplicate

```bash
python main.py run --path "/path/to/folder"
```

#### Deduplicate the indexed collection

```bash
python main.py dedup
```

#### Deduplicate and move duplicates

```bash
python main.py dedup --move "/path/to/duplicates"
```

The application keeps one canonical file from each similar-image group and moves the remaining files to the destination directory.

#### Generate embeddings

```bash
python main.py embed
```

Force regeneration:

```bash
python main.py embed --full
```

#### Cluster images

```bash
python main.py cluster-hdb
```

#### Semantic search

```bash
python main.py search "cat on a window"
```

Limit the number of results:

```bash
python main.py search "cat on a window" --top-k 10
```

#### Search by visual similarity

```bash
python main.py phash-search "/path/to/image.jpg"
```

Limit the number of results:

```bash
python main.py phash-search "/path/to/image.jpg" --top-k 10
```

#### Clear the local database

```bash
python main.py clear
```

> `clear` removes the local image index and stored application state. It does not delete your original image files.
## CLI commands

| Command        | Description                          |
| -------------- | ------------------------------------ |
| `ui-flet`      | Launch the desktop application       |
| `scan`         | Scan and index images                |
| `run`          | Scan and deduplicate in one workflow |
| `dedup`        | Find perceptually similar images     |
| `embed`        | Generate image embeddings            |
| `cluster-hdb`  | Cluster image embeddings             |
| `search`       | Semantic text-to-image search        |
| `phash-search` | Search visually similar images       |
| `clear`        | Clear the local database and index   |

## Project structure

```text
.
├── core/
│   ├── database.py
│   ├── scanner.py
│   ├── dedup.py
│   ├── embedder.py
│   ├── search.py
│   ├── clustererhdb.py
│   └── thumbnail_cache.py
├── screenshots/
├── app_flet.py
├── config.py
├── main.py
├── requirements.txt
├── install_and_run.command
├── install_and_run.bat
├── uninstall.command
├── uninstall.bat
├── README.md
└── README.en.md
```

## Local storage

The application keeps its local database and generated data under:

```text
storage/
```

The main SQLite database is:

```text
storage/images.db
```

The original image files are not copied into the database.

## Performance and hardware

The application is designed to work without a dedicated GPU.

### CPU

All major workflows can run on CPU, although embedding generation can be significantly slower for large collections.

### NVIDIA

When CUDA is available, PyTorch uses the NVIDIA GPU for embedding generation.

### Apple Silicon

On supported Apple Silicon systems, PyTorch uses Apple's MPS backend.

If available GPU memory is insufficient, the embedding implementation can fall back to CPU.

## Notes and limitations

* pHash is designed for perceptual similarity and may group images that are not byte-identical.
* The default pHash threshold is relatively strict (`6`) and can be adjusted in `config.py`.
* System and service directories are excluded by default when scanning.
* The desktop workflow automatically runs embeddings and clustering after scanning.
* The standard `requirements.txt` installs PyTorch, even though several CLI operations are implemented so that they can work without importing PyTorch.
* The first semantic-search/embedding run downloads the model from Hugging Face.
* The project is local-first, but the initial model download requires network access.
* Export copies the original files rather than modifying the source images.

## Configuration

Most processing parameters can be adjusted in `config.py`, including:

* supported image extensions;
* excluded directories;
* minimum file size;
* pHash threshold;
* embedding batch size;
* UMAP parameters;
* HDBSCAN parameters;
* semantic-search result limits.

## Development

The CLI entry point is:

```bash
python main.py --help
```

The processing pipeline is shared between the CLI and the Flet desktop application.

## License

Add the project's license here once a `LICENSE` file is included in the repository.

## Language

* [English](README.en.md)
* [Русский](README.md)
