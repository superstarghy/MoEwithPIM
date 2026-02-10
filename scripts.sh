#!/bin/bash

conda activate 3dcim

# prefilling and generating
python main.py --new_length 8 --prefill
python main.py --new_length 8 --kv_flag --prefill
python main.py --new_length 8 --go_flag --prefill
python main.py --new_length 8 --kv_flag --go_flag --prefill
python main.py --new_length 8 
python main.py --new_length 8 --kv_flag
python main.py --new_length 8 --go_flag
python main.py --new_length 8 --kv_flag --go_flag


# only generating
python main.py --new_length 8 
python main.py --new_length 8 --kv_flag --go_flag
python main.py --new_length 32 
python main.py --new_length 32 --kv_flag --go_flag
python main.py --new_length 64 
python main.py --new_length 64 --kv_flag --go_flag
python main.py --new_length 128 
python main.py --new_length 128 --kv_flag --go_flag

# new token = 1, no cache, only prefilling
python main.py --mvm_factor 1.0 --comm_factor 1.0 --group_factor 1.0 --prefix "1_" --new_length 1

python main.py --mvm_factor 0.4989 --comm_factor 3.9954 --group_factor 1.0 --prefix "U1C_" --new_length 1
python main.py --mvm_factor 0.7518 --comm_factor 3.9981 --group_factor 0.6 --prefix "U2C_" --new_length 1
python main.py --mvm_factor 0.5753 --comm_factor 3.9934 --group_factor 0.6 --prefix "S2C_" --new_length 1
python main.py --mvm_factor 1.2288 --comm_factor 3.9979 --group_factor 0.4 --prefix "U4C_" --new_length 1
python main.py --mvm_factor 1.0627 --comm_factor 3.9969 --group_factor 0.4 --prefix "S4C_" --new_length 1

python main.py --mvm_factor 0.4989 --comm_factor 1.8337 --group_factor 1.0 --prefix "U1O_" --new_length 1
python main.py --mvm_factor 0.7518 --comm_factor 2.3982 --group_factor 0.6 --prefix "U2O_" --new_length 1
python main.py --mvm_factor 0.5753 --comm_factor 3.0253 --group_factor 0.6 --prefix "S2O_" --new_length 1
python main.py --mvm_factor 1.2288 --comm_factor 2.8485 --group_factor 0.4 --prefix "U4O_" --new_length 1
python main.py --mvm_factor 1.0627 --comm_factor 3.6087 --group_factor 0.4 --prefix "S4O_" --new_length 1