# TF-QKD Optimization Framework

This project implements a high-performance optimization framework for Twin-Field Quantum Key Distribution (TF-QKD), focusing on different optical encodings:

- VSS (Vacuum-Single Superposition)
- CS (Coherent States)
- GS (General Gaussian States)

## Structure

src/tfqkd/       -> core physics + optimization  
scripts/         -> optimization runners  
data/            -> saved results  

## Setup

python -m venv venv
venv\Scripts\activate
pip install numpy scipy matplotlib pandas

## Run

python scripts/run_vss_q_optimization.py
python scripts/run_cs_b_optimization.py

## Output

Saved in:
data/optimization_results/<run_name>/

Includes:
- settings.json
- results csv
- plots

## Notes

- Parallel over loss points
- PSO-based optimization
- Spline continuation for dead zones

Author: Amirali Ekhteraei
