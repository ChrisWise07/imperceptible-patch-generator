#!/bin/bash
rm -rfd ./data/training_progress_data
mkdir ./data/training_progress_data
rm -rfd ./data/attack_training_data
mkdir ./data/attack_training_data
python3 -W ignore ./rrap_main.py