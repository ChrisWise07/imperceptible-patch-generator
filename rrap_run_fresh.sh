#!/bin/bash
rm -rfd ./experiment_data/test
python -W ignore ./code/main.py --data_folder_name 'test' --max_iter 1 --step_num 1 --patch_config "b" --decep_lr 0.00001 --percep_lr 0.0