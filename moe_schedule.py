import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse


# DATASET_NAME_MAP = {
#     "en_wikipedia": "Wiki",
#     "github": "GitHub",
#     "en_arxiv": "arXiv",
#     "en_book": "Book",
#     "en_cc": "CommonCrawl",
#     "en_c4": "C4",
#     "en_c4_ec": "C4E",
#     "en_stack": "Stack",
# }

def loadsum(load, group):
    load_sum = np.sum(load, axis=0) # shape 16
    result = []
    for group_idx in group:
        load_i = np.sum(np.take_along_axis(load_sum, group_idx, axis=0))
        result.append(load_i.item())
    return result


def single_token_schedule(gate_choices, group):
    """
    bandwith, buffer_size <= one token
    
    return:
        (np.array): (new_seq_len, #group), -1 means idle
    """
    seq_len = gate_choices.shape[0]
    load = []
    for group_idx in group:
        group_choices = gate_choices[:, group_idx]
        load.append(np.sum(group_choices, axis=1))
    load = np.array(load).T # (seq_len, #group)
    new_load = []
    for i in range(seq_len):
        while load[i].max() > 0:
            new_load.append(np.where(load[i]>0, i, -1))
            load[i] -= 1
    return np.array(new_load)


def compact_schedule(gate_choices, group):
    """
    unlimited bandwidth and buffer size
    
    return:
        (np.array): (new_seq_len, #group), -1 means idle
    """
    seq_len = gate_choices.shape[0]
    load = []
    for group_idx in group:
        group_choices = gate_choices[:, group_idx]
        load.append(np.sum(group_choices, axis=1))
    load = np.array(load) # (#group, seq_len)
    new_load = []
    max_len = 0
    for row in load:
        expand = np.repeat(np.arange(len(row)), row)
        new_load.append(expand)
        max_len = max(max_len, len(expand))
    out = np.full((len(new_load), max_len), -1, dtype=int)
    for i, r in enumerate(new_load):
        out[i, :len(r)] = r
    return out.T


def where_to_insert_space(res: np.array):
    indexes = []
    num_spaces = []
    # if not np.any(res > 0):
    #     return indexes, num_spaces
    idx = 0
    while(idx < len(res)):
        if res[-1] == 0: # reach max length after inserting spaces
            break
        if res[idx] > 0:
            # positive number
            window = res[idx:]
            num_pos = np.sum(window > 0)
            if num_pos > len(res) - idx - num_pos: # reuse opportunity
                # print(f"current res: {res}, idx={idx}")
                min_pos = np.min(window[window > 0])
                indexes.append(idx)
                num_spaces.append(min_pos)
                res[idx:] = res[idx:] - min_pos
                # print(f"after insert: {res}")
            assert res[-1] >= 0, "exceed max length"
        idx = idx + 1
    return np.array(indexes), np.array(num_spaces)


def optimized_compact_schedule(gate_choices, group):
    """
    gate_choices: (seq_len, num_experts)
    group: list of tuples, each tuple contains the indices of experts in that group
    return:
        (np.array): (schedule_length, #group), -1 means idle
    """
    # seq_len = gate_choices.shape[0]
    num_groups = len(group)
    load = []
    for group_idx in group:
        group_choices = gate_choices[:, group_idx]
        load.append(np.sum(group_choices, axis=1))
    load = np.array(load) # (#group, seq_len)
    max_id = np.argmax(np.sum(load, axis=1)) # the longest group
    max_len = np.sum(load[max_id])
    res = np.cumsum(load, axis=1) # cummulative sum of tokens
    # print("cumsum load:", res)
    # relative location information
    res = res[max_id] - res
    # print("relative location: ", res)

    compact_load = []
    for i in range(num_groups):
        expand = np.repeat(np.arange(len(load[i])), load[i])
        # insert spaces
        indexes, num_spaces = where_to_insert_space(res[i])
        # print(f"spaces indexes and number for group {i}: {indexes}, {num_spaces}")
        # insert
        for index, num_space in zip(indexes, num_spaces):
            if np.all(expand < index):
                break
            loc = np.argmax(expand >= index)
            expand = np.insert(expand, loc, [-1]*num_space)
        compact_load.append(expand)
    out = np.full((len(compact_load), max_len), -1, dtype=int)
    for i, r in enumerate(compact_load):
        out[i, :len(r)] = r
    return out.T


def count_load_sum(schedule):
    counts = [len(np.unique(row[row != -1])) for row in schedule]
    return np.sum(counts)

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="moe scheduler")
    parser.add_argument("--gate_load_path", type=str, default="./data/en_c4_ec_gate_choices.npy", help="moe workload traces directory")
    parser.add_argument("--num_layers", type=int, default=4, help="number of layers to be evaluated")
    args = parser.parse_args()
    gate_load_path = Path(args.gate_load_path)

    arr = np.load(gate_load_path) # (32, #token, 16)
    arr_sum = np.sum(arr, axis=1) # (32, 16)
    seq_length = arr.shape[1]
    print(f"trace: {gate_load_path}, shape: {arr.shape}")
    print(f"eg (layer 0): {arr[0, 0:, :]}")
    idx = np.argsort(arr_sum, axis=1) # (32, 16), group(i, 15-i)

    for layer_idx in range(args.num_layers): # layers
        print("="*35 + f" layer {layer_idx} " + "="*35)
        groups = {
            "U1": np.array([(i,) for i in range(16)]),
            "U2": np.array([(i, i+1) for i in range(0, 16, 2)]),
            "S2": np.array([(idx[layer_idx][i], idx[layer_idx][15-i]) for i in range(16 // 2)]),
            "U4": np.array([(i, i+1, i+2, i+3) for i in range(0, 16, 4)]),
            "S4": np.array([(idx[layer_idx][2*i], idx[layer_idx][2*i+1],
                    idx[layer_idx][15-2*i],idx[layer_idx][14-2*i]) 
                    for i in range(16 // 4)]),
        }
        for group_type, group in groups.items(): # grouping strategies
            batch_size = 512
            print(f"group {group_type}: {group.tolist()}")
            max_load = 0
            transfer1 = transfer2 = transfer3 = 0
            len_1 = len_2 = 0
            for i in range(0, len(arr), batch_size): # batch
                batch = arr[i:i+batch_size]
                workload = loadsum(batch[layer_idx], np.array(group))
                schedule1 = single_token_schedule(batch[layer_idx], np.array(group))
                schedule2 = compact_schedule(batch[layer_idx], np.array(group))
                schedule3 = optimized_compact_schedule(batch[layer_idx], np.array(group))
                
                batch_max_load = np.max(workload)
                batch_transfer1 = count_load_sum(schedule1)
                batch_transfer2 = count_load_sum(schedule2)
                batch_transfer3 = count_load_sum(schedule3)
                
                max_load += batch_max_load
                transfer1 += batch_transfer1
                transfer2 += batch_transfer2
                transfer3 += batch_transfer3
                len_1 += len(schedule1)
                len_2 += len(schedule2)
            # print(f"workload, max number: {max_load}")
            print(f"single token schedule: {len_1}({len_1 / seq_length}*4x), data transfer: {transfer1}({transfer1 / seq_length})")
            print(f"compact schedule: {len_2}({len_2 / seq_length}*4x), data transfer: {transfer2}({transfer2 / seq_length}), optimized: {transfer3}({transfer3 / seq_length})")
            print("-"*80)
    print("\n")

