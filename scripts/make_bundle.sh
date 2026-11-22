#!/usr/bin/env bash
# Build the Track 03 submission bundle: <TEAM>.zip with
#   estimated_trajectory.txt, metadata.json, src/ (main.cpp, main.py, ..., README.md)
# usage: scripts/make_bundle.sh CollegeName_TeamName
set -euo pipefail
team="${1:?usage: make_bundle.sh CollegeName_TeamName}"
[[ "$team" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "team name may only contain letters, digits, _ . -" >&2; exit 2; }
root="$(cd "$(dirname "$0")/.." && pwd)"
stage="$(mktemp -d)"
trap 'rm -rf -- "$stage"' EXIT
mkdir -p "$stage/$team/src"
cp -- "$root/submission/estimated_trajectory.txt" "$root/submission/metadata.json" "$stage/$team/"
cp -- "$root"/src/*.cpp "$root"/src/*.py "$root/src/README.md" "$stage/$team/src/"
cp -- "$root/requirements.txt" "$stage/$team/src/"  # build with the g++ command in src/README.md
rm -f -- "$root/$team.zip"
(cd "$stage" && zip -qr "$root/$team.zip" "$team")
echo "wrote $root/$team.zip"
