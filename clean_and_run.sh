#!/bin/bash

rm -rf venv
rm -rf data/processed/*
rm -rf results/*
rm -rf __pycache__/
rm -rf src/__pycache__/
rm -rf src/models/__pycache__/
rm -rf src/data/__pycache__/
rm -rf src/evaluation/__pycache__/
rm -rf src/utils/__pycache__/

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# python run_experiment.py
deactivate

echo "Done!"