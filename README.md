# Quran Shorts — daily automation

Renders one Quran short per day and posts it to YouTube. The renderer is a CPU
ray tracer (Rust); GitHub Actions builds it, fetches the recitation, renders,
uploads, and advances the progress record stored in this repo.

## What runs

`.github/workflows/daily.yml` (cron `0 12 * * *`, or manual dispatch):

1. `cargo build --release`
2. `python3 scripts/run_daily.py`
   - picks the next reciter + meaning-safe ayah range from `state/progress.json`
   - fetches that range from Backblaze B2 (`reciters/<reciter>/*.opus`), falling
     back to everyayah.com
   - picks a random texture (1K tiles → merged 4K) and a random SFX/seed
   - renders 1080×1920 @ 60 fps, CRF 12, all cores
   - picks a random Arabic title + description (`titling/`)
   - uploads to YouTube (same OAuth method as the old repo)
   - advances the progress and commits `state/progress.json`

## Order

- Reciters: the B2 list starting at Yasser, then Abdullaah, Abdurrahmaan, Ahmed,
  Ali, Aziz, Minshawy, Muhsin, Nasser, Sahl, Salah, Saood — then loop.
- Ranges: the **711** meaning-safe spans in `data/ranges.json`, in order.
- One successful post advances one range; after all 711, the next reciter starts.

## Secrets (same as the old repo)

| Secret | Purpose |
|---|---|
| `B2_AUTH_ID`, `B2_APP_KEY`, `B2_BUCKET_NAME` | Backblaze B2 (AWS-style names also accepted) |
| `YT_CLIENT_ID`, `YT_CLIENT_SECRET`, `YT_REFRESH_TOKEN` | YouTube OAuth |
| `YT_PRIVACY` (optional) | `public` / `unlisted` / `private` |

## Local run

```
pip install -r requirements.txt
cargo build --release
python3 scripts/run_daily.py
```

Set the same env vars (or a `.env`) before running. `short.mp4` and
`state/progress.json` are the outputs.

## Timings database

`assets/timings_db/<reciter>.json` — per-ayah word timings in reading order.

- Full per-word: Yasser (local), Abdurrahmaan, Minshawy, Saood (quran-align).
- Partial per-word: Muhsin (50 surahs), Abdullaah (10 surahs, quranicaudio).
- Ayah-timing only (one segment per ayah): the rest.

Every per-word time is validated against the ayah bounds and the Digital Khatt
spoken-word count; bad ayahs fall back to the ayah-long segment and are listed in
`assets/timings_db/suspicious.json`.

Regenerate with `python3 tools/build_timings_db.py`.

## Textures

`assets/textures/<name>/` holds 16× 1K tiles + `material.txt`. The workflow runs
`tools/merge_texture.py` to rebuild the 4K albedo before rendering (the merged
`.zip` is a build artifact, not committed).

## Layout

```
.github/workflows/daily.yml   daily job
scripts/run_daily.py          orchestrator
scripts/fetch_reciter.py      B2 / everyayah audio
scripts/upload_youtube.py     YouTube upload
scripts/titling.py            title/description
scripts/state.py              progress record
data/ranges.json              711 safe ranges
data/reciters.json            12 reciters (order, names, timings db)
data/surahs.json              114 surah names
titling/                      title + description templates
state/progress.json           last completed short
assets/timings_db/            hybrid word-timing database
assets/textures/              1K texture tiles
assets/sfx/                   background ambiences
src/                          CPU renderer
```