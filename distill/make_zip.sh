#!/bin/sh
# Build the single file to upload to Colab alongside pilot_colab.ipynb.
# Everything the notebook runs lives in rnakd/; the two protocol documents ride along so the run carries its own pre-registration.
set -e
cd "$(dirname "$0")"
rm -rf /tmp/rnakd_pilot rnakd_pilot.zip
mkdir -p /tmp/rnakd_pilot/rnakd_pilot
cp -r rnakd /tmp/rnakd_pilot/rnakd_pilot/
cp PILOT_PROTOCOL.md CLINICAL_PROTOCOL.md README.md /tmp/rnakd_pilot/rnakd_pilot/
find /tmp/rnakd_pilot -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
(cd /tmp/rnakd_pilot && zip -qr "$OLDPWD/rnakd_pilot.zip" rnakd_pilot)
rm -rf /tmp/rnakd_pilot
ls -lh rnakd_pilot.zip
unzip -l rnakd_pilot.zip
