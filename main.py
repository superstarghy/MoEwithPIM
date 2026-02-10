from threedsim.accelerator import Accelerator, AcceleratorConfig
from threedsim.inference import schedule_execution, fast_trace_decoder
from threedsim.models import DecoderOnlyTransformer
from threedsim.modules import TransformerDecoderLayer, MoELayer
from threedsim.modules.base import (
    assign_acc,
    fill_name_fields,
    make_traceable,
    make_use_linear,
)
from threedsim.mapping import Mapper, MapStrategy, Strategy
from threedsim.plotting import plot_graph

import matplotlib.pyplot as plt
import matplotlib as mpl
from adjustText import adjust_text

import os
import math
import argparse

# Set up argument parsing
parser = argparse.ArgumentParser(description="Simulator")
parser.add_argument('--mvm_factor', type=float, default=1.0, help='MVM factor')
parser.add_argument('--comm_factor', type=float, default=1.0, help='Communication factor')
parser.add_argument('--group_factor', type=float, default=1.0, help='group factor')
parser.add_argument('--prefix', default='', help='Prefix for the title')
parser.add_argument('--new_length', type=int, default=1, help='New length')
parser.add_argument('--kv_flag', action='store_true', help='Use KV cache')
parser.add_argument('--go_flag', action='store_true', help='Use GO cache')
parser.add_argument('--prefill', action='store_true', help='Use prefill')
parser.add_argument('--plot_trace', action='store_true', help='Plot the trace')
args = parser.parse_args()

mvm_factor = args.mvm_factor
comm_factor = args.comm_factor
prefix = args.prefix
group_factor = args.group_factor
new_length = args.new_length


bsz = 1
real_start_len = 32
real_target_len = real_start_len + new_length
if new_length == 1:
    start_len = math.ceil(real_start_len * mvm_factor)
    target_len = start_len + new_length
    prefill = True
else:
    start_len = real_start_len
    target_len = real_target_len
    prefill = args.prefill
num_layers = 1
d_model = 4096
d_ff = 688
vocab_size = 32000
kv_flag = args.kv_flag
go_flag = args.go_flag
device = "meta"
plot_trace = args.plot_trace

use_cache = go_flag # we can only manipulate mha
num_selects = 4
num_experts = 16
title = f"{prefix}{start_len}-{target_len}_d{d_model}_dff{d_ff}_{'kv' if kv_flag else ''}{'go' if go_flag else ''}{'' if prefill else '_noprefill'}"

# Configure the accelerator.
# NOTE: In the the accelerator class, you need to implement the functions that define the latencies [ns] and energy consumed [nJ] for every high-level operation we support.
# TODO: currently, if using go cache, buffer the whole gate and 'num_selects' outputs and set k=1
config = AcceleratorConfig(
    # tiles=100, tiers=1024, tier_shape=(512, 512),
    tiles=8192, tiers=1, tier_shape=(256, 256),
    num_digital_units=5, num_mha_units=4,
    mvm_latency=130, mvm_energy=12.5*group_factor,
    kv_caching=use_cache, kv_flag=kv_flag, go_flag=go_flag,
    moe_num_selects=num_selects, moe_num_experts=num_experts,
    # dram_bandwidth=44.8, dram_active_power=3.26, dram_inactive_power=0.625,
    mvm_factor=1.0, comm_factor=comm_factor,
)

# Create the accelerator
acc = Accelerator(config, device=device)
decoder_layer_kwargs = {
    "d_model": d_model,
    "nhead": 8,
    "dim_feedforward": d_ff,
}
embedding_layer_kwargs = {
    "vocab_size": vocab_size,
    "embedding_dim": d_model,
    "max_seq_length": target_len,
}
moe_kwargs = {
    "dim_feedforward": d_ff,
    "d_model": d_model,
    "k": num_selects,
    "num_experts": num_experts,
    "frequency": 1,
    # "density": torch.randn(3, 3).abs(),
}
# moe_layer = MoELayer(**moe_kwargs, device=device)

# Create the model
model = DecoderOnlyTransformer(
    TransformerDecoderLayer,
    num_layers=num_layers,
    decoder_layer_kwargs=decoder_layer_kwargs,
    embedding_layer_kwargs=embedding_layer_kwargs,
    moe_kwargs=moe_kwargs,
    device=device,
)

# Each model layer has access to the accelerator
assign_acc(model, acc)

# Create the mapper
mapper = Mapper(
    accelerator=acc,
    model=model,
    map_strategy=MapStrategy(
        strategy=Strategy.GREEDY_IN_ORDER, split_ffn=True, stack_embedding=True
    ),
)
mapper.map_network()
fill_name_fields(model)
make_traceable(model, is_traceable=True)

# Trace the model
make_use_linear(model, use_linear=True)
fast_traced = fast_trace_decoder(
    model, start_len=start_len, target_len=target_len, bsz=bsz, prefill=prefill
)

# Pipelined execution of the model
(
    execution_time,
    memory,
    peak_memory,
    energy,
    flops,
    energy_breakdown,
    latency_breakdown,
    mha_time,
    mha_energy
) = schedule_execution(
    fast_traced.graph,
    accelerator=model.accelerator,
    copy_and_cleanup_graph=False,
    communication=True,
    plot=plot_trace,
    plot_dir="results",
)
print(f"seq_len: {start_len} - {target_len}, d_model: {d_model}, d_ff: {d_ff}, kv: {kv_flag}, go: {go_flag}")
print(f"Execution took {execution_time} ns")
print(f"Required {peak_memory} bytes of scratchpad memory")
print(f"Spent {energy} nJ of energy")
print(f"FLOPs {flops}")

print("========== Energy Breakdown ==========")
energyBreak = {}
for label, value in energy_breakdown.items():
    if label == 'mha':
        energyBreak['dram_kv'] = value['dram_kv']
        # energyBreak['dram_go'] = value['dram_go']
        energyBreak['mha-comp'] = value['comp']
    else:
        energyBreak[label] = value
print(energyBreak)

print("========== Latency Breakdown ==========")
LatencyBreak = {}
for label, value in latency_breakdown.items():
    if label == 'mha':
        LatencyBreak['dram_kv'] = value['dram_kv']
        # LatencyBreak['dram_go'] = value['dram_go']
        LatencyBreak['mha-comp'] = value['comp']
    else:
        LatencyBreak[label] = value
print(LatencyBreak)

path = f"results/{real_start_len}-{real_target_len}_d{d_model}_dff{d_ff}/{title}.txt"
os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path, "w") as f:
    f.write(f"seq_len: {start_len} - {target_len}, d_model: {d_model}, \
            d_ff: {d_ff}, kv: {kv_flag}, go: {go_flag}, mvm_factor: {mvm_factor}, comm_factor: {comm_factor}\n")
    f.write(f"Execution took {execution_time} ns\n")
    # f.write(f"Memory usage: {memory} bytes\n")
    f.write(f"Required {peak_memory} bytes of scratchpad memory\n")
    f.write(f"Spent {energy} nJ of energy\n")
    f.write(f"FLOPs {flops}\n")
    f.write(f"MHA time {mha_time}, ({execution_time - mha_time[-1]})\n")
    f.write(f"MHA energy {mha_energy}\n")
    f.write("========== Energy Breakdown ==========\n")
    f.write(str(energyBreak) + "\n")
    f.write("========== Latency Breakdown ==========\n")
    f.write(str(LatencyBreak) + "\n")

def plotbreakdown(data, filename='results.svg', threshold=1):
    labels = data.keys()
    sizes = data.values()
    cmap = mpl.colormaps.get_cmap("tab20").resampled(len(labels))
    color_map = {label: cmap(i) for i, label in enumerate(sorted(set(labels)))}
    colors = [color_map[l] for l in labels]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, _, autotexts = ax.pie(
        sizes, labels=None,
        startangle=90,
        colors=colors,
        autopct=lambda pct: ('%1.1f%%' % pct) if pct > threshold else '',
        pctdistance=0.7,
    )
    ax.set_aspect('equal')
    
    texts = []
    for autotext in autotexts:
        # autotext.set_color("black")
        texts.append(autotext)
    # adjust_text(texts, arrowprops=dict(arrowstyle="-", color='black'))
    adjust_text(texts)
    
    plt.legend(labels, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)


plotbreakdown(energyBreak, f"results/{real_start_len}-{real_target_len}_d{d_model}_dff{d_ff}/Energy_{title}.svg")
plotbreakdown(LatencyBreak, f"results/{real_start_len}-{real_target_len}_d{d_model}_dff{d_ff}/Latency_{title}.svg")

# Plot
print("========== Plot (in results/) ==========")
# fast_traced = fast_trace_decoder(
#     model, start_len=start_len, target_len=target_len, bsz=bsz
# )
if plot_trace:
    plot_graph(fast_traced, f"results/{real_start_len}-{real_target_len}_d{d_model}_dff{d_ff}/fast_tracing.svg")

# (
#     execution_time,
#     memory,
#     peak_memory,
#     energy,
#     flops,
#     energy_breakdown,
#     latency_breakdown,
# ) = schedule_execution(
#     fast_traced.graph,
#     accelerator=model.accelerator,
#     copy_and_cleanup_graph=False,
#     plot=True,
#     plot_dir="results",
#     communication=True,
# )