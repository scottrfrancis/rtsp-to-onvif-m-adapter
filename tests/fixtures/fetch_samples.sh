#!/usr/bin/env bash
# Fetch public test assets for the detector accuracy tests. NOT committed
# (gitignored binaries) — run this once to populate fixtures.
#
#   bash tests/fixtures/fetch_samples.sh
#   pytest tests/test_detect_accuracy.py
#
# COCO val2017 image_id 785 (000000000785.jpg) — a single-person reference image
# with a published ground-truth person box. Image © its Flickr owner, served by
# the COCO dataset; used here for evaluation only (download-on-demand).
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$DIR/coco"
IMG="$DIR/coco/000000000785.jpg"

if [[ ! -f "$IMG" ]]; then
  echo "Downloading COCO val2017 000000000785.jpg ..."
  curl -sSL -o "$IMG" http://images.cocodataset.org/val2017/000000000785.jpg
fi
echo "fixtures ready: $IMG"
