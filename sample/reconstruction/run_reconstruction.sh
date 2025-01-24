#!/bin/bash

mkdir normal_distribution_data
mkdir normal_distribution_data/cov
mkdir normal_distribution_data/before
mkdir normal_distribution_data/after

python3 create_cov.py

python3 create_MPS.py

ft normal_distribution_init.yml
ft normal_distribution.yml

python3 visualize_edges.py