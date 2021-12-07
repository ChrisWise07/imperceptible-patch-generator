#!/bin/bash
rm -rfd ./experiment_data/test1/test2
python -W ignore ./main.py --data_folder_name 'test1' --max_iter 1 --step_num 1 #--previous_experiment_directory_name 'test'