#!/bin/bash
# Export Tabernacle Review Artifacts to Desktop

DEST="${HOME}/Desktop/Tabernacle_Architecture_Review"
SRC="/Users/enos/.gemini/antigravity/brain/8625ed10-93b5-4353-8ed9-e34d96580592"

echo "Exporting artifacts to: $DEST"
mkdir -p "$DEST"

cp "$SRC/architectural_review.md" "$DEST/"
cp "$SRC/dependency_graph.mermaid" "$DEST/"
cp "$SRC/health_check.md" "$DEST/"
cp "$SRC/walkthrough.md" "$DEST/"
cp "$SRC/task.md" "$DEST/"

echo "✅ Export complete."
ls -l "$DEST"
open "$DEST"
