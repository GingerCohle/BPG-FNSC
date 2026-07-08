#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-${DATA_ROOT:-/media/kejunjie_coco/autohash}}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p "${REPO_ROOT}/data"

link_one() {
  local name="$1"
  local target="$2"
  if [[ ! -d "${target}" ]]; then
    echo "Missing dataset directory: ${target}" >&2
    return 1
  fi
  ln -sfn "${target}" "${REPO_ROOT}/data/${name}"
  echo "linked data/${name} -> ${target}"
}

link_one activitynet "${DATA_ROOT}/ActivityNet"
link_one fcv "${DATA_ROOT}/fcv"
link_one hmdb "${DATA_ROOT}/hmdb"
link_one ucf "${DATA_ROOT}/ucf"

