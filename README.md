# Quran Shorts — daily automation

Renders one reviewed Quran short per day and posts it to YouTube. The renderer
is the CPU ray tracer in `src/`; the review app lives one directory up in
`App/`.

The app is the only place where timings are edited. It writes a self-contained
folder:

```
App/exports/<n>/
  audio.mp3    192 kbps crop of the selected safe range
  short.json   title, description, reciter/surah/range metadata and word keyframes
```

Upload that folder (unchanged) to Backblaze B2 as `recit/<n>/`. This repo never
edits timings and never joins per-ayah audio; it downloads the folder and
renders it directly.

## What the daily job does

`.github/workflows/daily.yml` (cron `0 12 * * *`, or manual dispatch):

1. builds the Rust renderer (`cargo build --release`)
2. runs `scripts/run_daily.py`
   - lists the numeric folders under `recit/` on B2
   - starts after `last_folder` in `state/progress.json`
   - renders the next folder with `lumen3d --short-json <folder>`
   - picks a random texture (1K tiles → merged 4K) and a random SFX file
   - uploads the rendered `short.mp4` to YouTube using the title/description
     stored in the folder's `short.json`
   - saves `last_folder` and commits `state/progress.json`
3. uploads `short.mp4` as a workflow artifact

## Folder order

Folders are numeric and the order is a round-robin over the folders that exist
at run time:

- after the last folder, the next higher folders render in order
- newly added higher folders are therefore picked up in order
- after the highest folder, it wraps to the lowest one

Examples:

| existing folders | last_folder | next |
|---|---|---|
| 1..10 | 4 | 5 |
| 1..10 | 10 | 1 |
| 1..10, then add 11 | 10 | 11 |
| 1..10, then add 11 | 11 | 1 |

A folder that is missing, invalid (`short.json`/`audio.mp3` unreadable, bad
word order, times outside the audio, not reviewed) or fails to render is
logged, skipped, and `last_folder` advances past it. An **upload failure does
not advance**, so the same folder is retried on the next run.

## `short.json` contract

The renderer reads:

- `surah.number`, `range.from`, `range.to` — selects the DigitalKhatt text
- `words[]` — `text_index` (0-based spoken-word index in the range),
  `start`, `end` in seconds **relative to `audio.mp3`**
- `audio.file`, `audio.duration`
- `title`, `description` — one choice
- `titles`, `descriptions` — optional lists of ~10 variations (from `GitHub/titling/`);
  when present, the renderer picks **any random one** each run (so a looping
  schedule posts a different wording each time); falls back to `title`/`description`.

Repeated `text_index` values are allowed and expected: a word recited twice is
two keyframes pointing at the same Quran word. The old `--range` mode remains
available for debugging, but the daily job only uses `--short-json`.

## Secrets

| Secret | Purpose |
|---|---|
| `B2_AUTH_ID`, `B2_APP_KEY` **or** `B2_APPLICATION_KEY`, `B2_BUCKET_NAME` | Backblaze B2 (`AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `BUCKET_NAME` also accepted; bucket default `main-one`) |
| `YT_CLIENT_ID`, `YT_CLIENT_SECRET`, `YT_REFRESH_TOKEN` | YouTube OAuth |
| `YT_PRIVACY` (optional) | `public` / `unlisted` / `private` (default `public`) |
| `YT_CATEGORY_ID` (optional) | YouTube category id (default `22`) |

These are the **same secret names as the original `GitHub` repo**, so an existing
setup keeps working when this folder replaces it; no renaming of secrets.

The B2 key needs list + read access to the bucket. The app uploads folders
manually; this repo only reads `recit/`.

## Local run

```bash
pip install -r requirements.txt
cargo build --release

# Render one local export without B2 or YouTube:
python3 scripts/run_daily.py \
  --source-dir ../App/exports \
  --folder 1 \
  --skip-upload \
  --no-sfx
```

`--skip-upload` stops after a successful render and leaves
`state/progress.json` unchanged. Without `--source-dir`, the script authorizes
against B2 using the environment variables above.

A full production render is 1080×1920 @ 60 fps, CRF 12. The workflow has a
360-minute timeout.

## Textures and SFX

- `assets/textures/<name>/` holds 16× 1K tiles + `material.txt`. The job runs
  `tools/merge_texture.py` to build the 4K albedo zip before rendering (the zip
  is a build artifact, not committed).
- `assets/sfx/` holds background ambiences. The renderer picks one from the
  render seed. Files longer than the short start at a deterministic random
  offset; shorter files are looped with a crossfade.

## Layout

```
.github/workflows/daily.yml   daily job
scripts/run_daily.py          folder round-robin + render + upload
scripts/b2.py                 B2 list/download for recit/<n>/
scripts/state.py              last_folder progress
scripts/upload_youtube.py     YouTube upload
data/                         surah names + legacy range data
assets/textures/              1K texture tiles
assets/sfx/                   background ambiences
src/                          CPU renderer
state/progress.json           last posted/skipped folder
```
