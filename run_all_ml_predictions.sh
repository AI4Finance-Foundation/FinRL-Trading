#!/usr/bin/env bash
set -e

# Ensure we're in the right directory
cd "$(dirname "$0")"

# Activate the virtual environment
source venv/bin/activate

echo "=========================================================="
echo "    Running ML Stock Selection Pipeline (3 Key Tasks)     "
echo "=========================================================="

echo -e "\n[1/3] Generating 2025-09-30 Selection..."
python src/strategies/ml_bucket_selection.py --val-cutoff 2025-06-30 --infer-date 2025-09-30
echo "Saved prediction to outputs/selection/ directory."

echo -e "\n[2/3] Generating 2026-03-31 Selection..."
python src/strategies/ml_bucket_selection.py --val-cutoff 2025-12-31 --infer-date 2026-03-31
echo "Saved prediction to output/selection/ directory."

echo -e "\n[3/3] Generating LATEST LIVE PREDICTION (Mixed Vintage)..."
python src/strategies/ml_bucket_selection.py --val-cutoff 2025-12-31 --mixed-vintage
echo "Saved prediction to output/selection/ directory."

echo -e "\n=========================================================="
echo "All done! Check the output/selection/ directory for the CSV files."
echo "=========================================================="
