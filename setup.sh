#!/bin/bash
set -e

# Install the main requirements
pip install -r requirements.txt

# Ensure frequently missed tools are present.  These are needed for
# the groove sampler utilities and other scripts.  If they are already
# installed the commands below are a no-op.
python - <<'EOF'
import importlib.util, subprocess, sys
missing = [pkg for pkg in ("pretty_midi", "tqdm")
           if importlib.util.find_spec(pkg) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
EOF

