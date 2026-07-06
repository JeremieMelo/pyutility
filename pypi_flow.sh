#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./update_pypi.sh testpypi
#   ./update_pypi.sh pypi
#
# Assumes:
#   - pyutils/version.py contains: __version__ = "..."
#   - ~/.pypirc already defines [testpypi] and/or [pypi]
#   - python -m pip install build twine
#
# This script:
#   1) reads the version from pyutils.version.__version__
#   2) cleans old build artifacts
#   3) builds sdist + wheel
#   4) checks metadata
#   5) uploads to the selected repository
#   6) creates a temporary venv
#   7) installs the package from that repository
#   8) verifies import + pip check
#   9) removes the temp venv

REPO="${1:-}"
if [[ -z "${REPO}" ]]; then
  echo "Usage: $0 <testpypi|pypi>"
  exit 1
fi

if [[ "${REPO}" != "testpypi" && "${REPO}" != "pypi" ]]; then
  echo "Repository must be 'testpypi' or 'pypi'"
  exit 1
fi

PKG_NAME="torchonn-pyutils"
IMPORT_NAME="pyutils"
VERSION_FILE="pyutils/version.py"

if [[ ! -f "${VERSION_FILE}" ]]; then
  echo "Version file not found: ${VERSION_FILE}"
  exit 1
fi

VERSION="$(python - <<'PY'
from pathlib import Path
import re

text = Path('pyutils/version.py').read_text()
m = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', text)
if not m:
    raise SystemExit('Could not find __version__ in pyutils/version.py')
print(m.group(1))
PY
)"

echo "========================================================"
echo "Package : ${PKG_NAME}"
echo "Version : ${VERSION}"
echo "Target  : ${REPO}"
echo "========================================================"

echo "== Clean build artifacts =="
rm -rf build dist *.egg-info

echo "== Build =="
python -m build

echo "== Twine check =="
twine check dist/*

echo "== Wheel metadata =="
unzip -p dist/*.whl '*.dist-info/METADATA' | grep '^Requires-Dist' || true

echo "== Upload to ${REPO} =="
twine upload -r "${REPO}" dist/*

echo "== Create temp environment =="
TMPDIR="$(mktemp -d)"
VENV_DIR="${TMPDIR}/venv"
python -m venv "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python -m pip install -U pip

if [[ "${REPO}" == "testpypi" ]]; then
  INSTALL_ARGS=(--index-url https://test.pypi.org/simple --extra-index-url https://pypi.org/simple)
else
  INSTALL_ARGS=()
fi

echo "== Install from ${REPO} =="
python -m pip install "${INSTALL_ARGS[@]}" "${PKG_NAME}"

echo "== Verify import =="
python - <<PY
import ${IMPORT_NAME}
print('import ok:', ${IMPORT_NAME}.__file__)
PY

echo "== pip check =="
python -m pip check

deactivate

echo "== Clean temp environment =="
rm -rf "${TMPDIR}"

echo "All done."
