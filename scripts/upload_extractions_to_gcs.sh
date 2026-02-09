#!/usr/bin/env bash
# Upload local extractions/ to GCS for Cloud Run deployment.
#
# Usage:
#   ./scripts/upload_extractions_to_gcs.sh [BUCKET_NAME]
#
# Default bucket: g1-experiment-videos

set -euo pipefail

BUCKET="${1:-g1-experiment-videos}"
SRC_DIR="$(cd "$(dirname "$0")/.." && pwd)/extractions"

if [ ! -d "$SRC_DIR" ]; then
  echo "Error: extractions/ directory not found at $SRC_DIR"
  exit 1
fi

echo "Uploading extractions/ to gs://$BUCKET/extractions/"
echo "Source: $SRC_DIR"
echo ""

# Use gsutil rsync for efficient incremental uploads
# -r: recursive, -c: compare checksums (skip unchanged files)
gsutil -m rsync -r -c "$SRC_DIR" "gs://$BUCKET/extractions"

echo ""
echo "Upload complete. Verify with:"
echo "  gsutil ls gs://$BUCKET/extractions/"
