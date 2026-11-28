#!/bin/sh
# Optional: AVX2/FMA build (same pixels, ~8-13% faster). Needs a CPU from ~2013+ (Intel Haswell / AMD Zen).
RUSTFLAGS="-C target-cpu=x86-64-v3" cargo build --release
