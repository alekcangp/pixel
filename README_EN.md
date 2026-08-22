**English** | [Русский](README.md)

# 📸 Pixel — smart collector and organizer for your photos

Pixel is a simple app for anyone whose photos are scattered all over the computer: different drives, USB sticks, folders.

The app does most of the work for you: it **finds photos, removes duplicates and junk, then groups images by subject**.

All that's left for you is to quickly review the result and pick the best shots.

---

## ✨ Features

- 🧲 **Collects everything from everywhere** — can automatically scan all available drives and find photos. No need to open folders one by one.
- 🤖 **Puts things in order automatically** — finds identical and visually similar images, so you don't have to sort through numerous copies by hand.
- 🗂️ **Sorts by subject** — a neural network groups visually similar photos into categories: nature, documents, animals, holidays, travel, etc.
- 👀 **Fast visual review** — a convenient gallery for browsing and picking the best frames.
- 🔍 **Search** — semantic search by description ("cat on the couch") and search for similar images using an example photo.
- 🌐 **Interface in English and Russian** — the language is selected automatically based on system settings.

---

# 🚀 Getting started in 3 steps

## Step 1. Download

1. Open the project page: **https://github.com/alekcangp/pixel**
2. Click **Code → Download ZIP**.
3. Unpack the archive to any convenient location.

## Step 2. Run

- **Windows:** double-click **`pixel.bat`**
- **macOS:** run **`pixel.command`**

On first launch, Pixel will automatically install everything needed and download the AI model. This only happens once.

> 💡 On macOS, you may need to allow the first launch: right-click `pixel.command` → "Open".

## Step 3. Wait for the first scan

The first scan takes the longest: from a few minutes to several hours. It depends on the number of photos, data volume, disk speed and CPU.

Progress is shown right in the **Pixel window title**. You can close the app — already processed data is saved in the database, and after relaunch work continues where it left off.

All subsequent launches are fast: Pixel remembers files and processes only new or changed ones.

![Pixel main window](screenshots/main.png)

---

## 🖱️ Gallery controls

**With the mouse:**

- **Left click** on a photo — select/deselect; in fullscreen view — previous / next photo.
- **Right click** — open/close fullscreen view;
- **Mouse wheel** — scroll the gallery; in fullscreen view — smooth zoom in/out.

**Keyboard shortcuts (in fullscreen view):**

| Key | Action |
|---|---|
| `←` / `→` | previous / next photo |
| `+` / `-` | zoom in / out |
| `0` | reset zoom |
| mouse drag | pan the zoomed photo |
| `Esc` | close the viewer |

---

## ❓ FAQ

**What happens to my photos?**

Nothing bad. During scanning, Pixel **does not move or delete any files** — it only builds its own local index. The originals stay exactly where they are.


**Is it safe?**

Pixel runs fully locally. Photos are never uploaded anywhere. The only thing downloaded from the internet is the AI model (once); after that it works on your computer without a network.



**How do I uninstall Pixel?**

- Windows: `pixel.bat uninstall`
- macOS: `./pixel.command uninstall`

This removes the app, its database, and the downloaded model. Your photos are not affected.

---

# 🛠️ Developer section

## Tech stack

| Component | Technology |
|---|---|
| Language | Python 3.10+ (macOS / Windows) |
| GUI | Flet |
| Embeddings | PyTorch + Transformers (SigLIP `google/siglip-base-patch16-224`) |
| Vector search | FAISS |
| Clustering | UMAP-learn + HDBSCAN |
| Storage | SQLite (metadata, embeddings, thumbnails, clusters) |
| Images | Pillow, imagehash (perceptual hash) |


## Architecture

```
main.py               CLI entry point (argparse, subcommands)
app_flet.py           Flet GUI + pipeline orchestration
config.py             All settings (paths, thresholds, UMAP/HDBSCAN params)
i18n.py               ru/en localization ("auto" — based on OS locale)
core/
  scanner.py          Directory scanning, incremental updates
  database.py         SQLite storage
  dedup.py            pHash + LSH index + Union-Find
  embedder.py         SigLIP embeddings
  clustererhdb.py     UMAP + HDBSCAN (+ mega-cluster refinement, KNN noise rescue)
  search.py           Semantic search (FAISS) and pHash-based search
  thumbnail_cache.py  WebP thumbnails stored in SQLite
```

Pipeline: **scan → dedup → embed → cluster**, then browsing/search in the UI. Every step is incremental by default; the `--full` flag recomputes everything from scratch.

## Manual installation

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The SigLIP model is downloaded by the launcher during installation (or automatically on first use of the embedder if installed manually). Cache location: `~/.cache/huggingface/hub/` (Windows: `%USERPROFILE%\.cache\huggingface\hub`).

## Running

```bash
python main.py ui-flet                 # desktop GUI
python main.py --help                  # list of all commands
```

The launchers support commands: `install` (install only), `run` (run only), `uninstall`.

### CLI examples

```bash
python main.py scan --path ~/Photos                # index a folder
python main.py run --path ~/Photos                 # scan + dedup in one pass
python main.py embed                               # SigLIP embeddings
python main.py cluster-hdb                         # UMAP+HDBSCAN clustering
python main.py search "sunset over the sea"        # semantic text search
python main.py phash-search path/to/img.jpg        # find similar by pHash
python main.py clear                               # fully clear storage/
```

Common options for scan/run/dedup: `--ext` (extensions), `--exclude` (excluded dirs, added to defaults), `--min-size` (min file size in KB), `--move <dir>` (move duplicates), `--full` (full recompute instead of incremental).


## Configuration (`config.py`)

All parameters live in one file, `config.py`, with detailed comments next to each setting.

Key parameters:

| Parameter | Default | What it does |
|---|---|---|
| `DEFAULT_EXTENSIONS` | jpg, jpeg, png, webp, bmp, gif, tiff | which files count as images |
| `MIN_FILE_SIZE` | 30 KB | smaller files are skipped |
| `PHASH_THRESHOLD` | 6 | max Hamming distance for merging duplicates (lower = stricter) |
| `SIGLIP_MODEL` | `google/siglip-base-patch16-224` | embedding model |
| `EMBED_BATCH_SIZE` | 16 | images processed per model pass |
| `UMAP_*`, `HDBSCAN_*`, `SUB_*` | see `config.py` | subject clustering parameters |
| `SEARCH_TOP_K`, `SEARCH_THRESHOLD` | 500 / 0.05 | semantic search: result count and relevance threshold |
| `PREWARM_THUMB_WORKERS` | 4 | background threads for thumbnail generation |
| `APP_LANGUAGE` | `"auto"` | UI language: `ru`, `en`, or auto-detect |

---

> Note: the Russian version of this document is available in [README.md](README.md). | Русская версия документа доступна в [README.md](README.md).
