#!/bin/bash
set -eu
mkdir -p /logs/verifier
printf '0\n' > /logs/verifier/reward.txt
python3 /tests/verify.py
printf '1\n' > /logs/verifier/reward.txt
