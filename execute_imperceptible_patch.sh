#!/bin/bash
rm -rfd ./experiment_data/test
python -W ignore ./code/main.py --data_folder_name 'test' --max_iter 1000 --step_num 50