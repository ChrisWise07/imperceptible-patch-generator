#!/bin/bash
rm -rfd ./experiment_data/test1
python -W ignore ./code/main.py --data_folder_name 'test1' --max_iter 1 --step_num 1 --patch_config "is" #--previous_experiment_directory_name 'test'