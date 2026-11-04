#!/usr/bin/env bash
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
git clone --depth 1 https://github.com/agossouema2011/WCE2024.git third_party/WCE2024
git -C third_party/WCE2024 apply "$ROOT/patches/wce2024_yolov5_compat.patch"
echo "WCE2024 ready. Download yolov5s.pt into third_party/WCE2024/Yolov5/yolov5s.pt"
