import math
from functools import partial
from typing import Callable
from functools import lru_cache

import torch
from torch import Tensor

from .utils import get_logger


def latency_table(op, d_model, seq_length):
    """
    ns for one token, attention relates to seq_length.
    
    MVM: 100,
    MHA: n^2 (alpha * d_model + beta * d_model^2),
    LayerNorm: ceil(d_model/512)*50,
    RELU: 1,
    GeLU: 16*(d_model/256)+6,
    Add: 8*(d_model/256)+4,
    Communication: ceil(d_model/512)*24,
    """
    # d_model = 256, 512, 768, 1024
    # seq_length = 64, 128, 512
    table = { # from 3dcim paper
        "MVM": [100, 100, 100, 100], # 100
        "MHA64": [4295, 16775, 37575, 66695], # 0.0635x^2+135
        "MHA128": [17031, 66567, 149127, 264711], # 4(0.0630x^2+129.75)
        "MHA512": [270855, 1058823, 2372103, 4210695], # 64(0.0626x^2+128.11)
        "LayerNorm": [50, 50, 100, 100],
        "ReLU": [1, 1, 1, 1],   # 1
        "GeLU": [22, 38, 54, 70], # y=16 * x/256 + 6
        "Add": [12, 20, 28, 36],    # y=8 * x/256 + 4
        "Communication": [24, 24, 48, 48],
    }
    
    alpha = 1.064e-04
    beta = 1.521e-05
    if op == "MVM":
        latency = 100
    elif op == "MHA":
        # latency = (seq_length / 64) ** 2 * int(d_model ** 2 * 0.0635 + 135)
        latency = seq_length ** 2 * (alpha * d_model + d_model ** 2 * beta)
    elif op == "LayerNorm":
        latency = (d_model / 256) * 50
    elif op == "ReLU":
        latency = 1
    elif op == "GeLU":
        latency = (16 * (d_model / 256) + 6)
    elif op == "Add":
        latency = (8 * (d_model / 256) + 4)
    elif op == "Communication":
        latency = (d_model / 256) * 12
    else:
        raise ValueError(f"Unknown operation: {op}")
    return math.ceil(latency) # round to nearest ns


def energy_table(op, d_model, seq_length):
    """
    nJ for one token, attention relates to seq_length.
    
    MHA: gamma * d_model * seq_length^2,
    """
    # d_model = 256, 512, 768, 1024
    # seq_length = 64, 128, 512
    table = { # pJ, from 3dcim paper
        "MVM": [10_000, 10_000, 10_000, 10_000],
        "MHA64": [23_376.37, 46_743.98, 70_111.60, 93_483.50], # 91.28x+6.61
        "MHA128": [93_496.90, 186_971.66, 280_466.41, 373_925.45], # 4(91.2875x+5)
        "MHA512": [1_495_907.65, 2_991_525.13, 4_487_142.60, 5_982_764.36], # 64(91.2853x+4.5)
        "LayerNorm": [20.96, 20.96, 41.92, 41.92],
        "ReLU": [0.02, 0.03, 0.05, 0.06],
        "GeLU": [9.63, 19.25, 28.88, 38.51],
        "Add": [9.43, 18.86, 28.29, 37.72],
        "Communication": [12.93, 25.87, 38.80, 51.73],
    }
    gamma = 0.022289 * 25 # from the supplementary
    if op == "MVM":
        energy = 10000
    elif op == "MHA":
        energy = (seq_length ** 2) * gamma * d_model
    elif op == "LayerNorm":
        energy = (d_model / 256) * 20.96
    elif op == "ReLU":
        energy = (d_model / 256) * 0.02
    elif op == "GeLU":
        energy = 9.63 * (d_model / 256)
    elif op == "Add":
        energy = 9.43 * (d_model / 256)
    elif op == "Communication":
        energy = 12.93 * (d_model / 256)
    else:
        raise ValueError(f"Unknown operation: {op}")
    return energy / 1000 # convert to nJ


def communication_latency(d_model: int, input_or_not: bool, seq_length: int, dram_bandwidth, factor=1.0):
    # assume 8 bits
    # d_model = 512 # every tile is the same
    # if input_or_not:
    #     latency = seq_length * d_model / dram_bandwidth * factor
    # else:
    #     latency = seq_length * d_model / dram_bandwidth
    # return math.ceil(latency) / 10 # ns
    if input_or_not:
        return latency_table("Communication", d_model, seq_length) * factor
    else:
        return latency_table("Communication", d_model, seq_length)
    
def communication_energy(d_model: int, input_or_not: bool, seq_length: int, dram_bandwidth, dram_active_power, factor=1.0):
    # energy = communication_latency(d_model, input_or_not, seq_length, dram_bandwidth, factor) * dram_active_power # nJ
    # return energy
    if input_or_not:
        return energy_table("Communication", d_model, seq_length) * factor
    else:
        return energy_table("Communication", d_model, seq_length)

def kv_cache_latency(vector_size: int, seq_length: int, dram_bandwidth: float) -> float:
    """
    Calculate the latency (ns) for loading the KV cache in nano seconds.
    We assume dram_bandwidth GB/s bandwidth.
    """
    # one for key, one for value
    # we assume the KV-cache can be quantized to 8-bit
    num_bytes_in_cache = 2 * seq_length * vector_size
    time_to_load = 1e9 * ((num_bytes_in_cache / 1e9) / dram_bandwidth)
    return math.ceil(time_to_load)  # round up to the nearest ns


def go_cache_latency(
    vector_size: int, seq_length: int,
    num_selects: int, num_experts: int,
    dram_bandwidth: float
) -> float:
    """
    Calculate the latency (ns) for loading the GO cache in nano seconds.
    TODO: Currently, we only buffer the whole gate and k outputs. Assume 8 bits
    Assume dram_bandwidth GB/s bandwidth.
    """
    # one for gate, one for output
    # num_bytes_in_cache = seq_length * num_experts + seq_length * num_selects * vector_size
    num_bytes_in_cache = seq_length * num_experts + num_experts * vector_size
    time_to_load = 1e9 * ((num_bytes_in_cache / 1e9) / dram_bandwidth)
    return math.ceil(time_to_load)  # round up to the nearest ns


@lru_cache(maxsize=4)
def calculate_mha_latency(
    vector_size: int,
    seq_length: int,
    causal: bool,
    num_heads: int,
    prefill: bool,
    kv_flag: bool,
    go_flag: bool,
    num_selects: int,
    num_experts: int,
    model_sram: bool,
    dram_bandwidth: int,
):
    """
    Calculates the latency for doing multi-head-attention
    in nano seconds.

    Args:
        vector_size (int): Length of the input vectors (d_model).
        seq_length (int): Sequence length (number of tokens).
        causal (bool): If causal attention is used. Essentially halves the number of ops.
        num_heads (int): Number of heads for the attention.
        kv_caching (bool): If we are using KV-caching.
        model_sram (bool): If we are using SRAM in the model.
        dram_bandwidth (int): Bandwidth of the DRAM in GB/s.

    Returns:
        (float, float): KV-cache latency in ns, MHA latency in ns.
    """
    #? model_sram, num_heads
    # if model_sram:
    #     return 0, latency_table("MHA", vector_size, seq_length)
    # TODO: if kv_caching==False and go_flag==True: prefilling
    if kv_flag and not prefill: # only one token
        _mha_latency = latency_table("MHA", vector_size, seq_length) / seq_length
        _kv_cache_latency = kv_cache_latency(vector_size, seq_length, dram_bandwidth)
    else:
        _mha_latency = latency_table("MHA", vector_size, seq_length)
        _kv_cache_latency = 0
    tri = 0.5 if causal else 1.0
    _mha_latency = _mha_latency * tri

    if go_flag and not prefill:
        _go_cache_latency = go_cache_latency(vector_size, seq_length, num_selects, num_experts, dram_bandwidth)
    else:
        _go_cache_latency = 0
    return _go_cache_latency, _kv_cache_latency, _mha_latency


@lru_cache(maxsize=4)
def calculate_mha_energy(
    vector_size: int,
    seq_length: int,
    causal: bool,
    prefill: bool,
    kv_flag: bool,
    go_flag: bool,
    num_selects: int,
    num_experts: int,
    model_sram: bool,
    dram_bandwidth: float,
    dram_active_power: float,
):
    """
    Calculate the energy consumed by performing multi-head attention.

    Args:
        vector_size (int): Length of the input vectors (d_model).
        seq_length (int): Sequence length (number of tokens).
        causal (bool): If causal attention is used. Essentially halves the number of ops.
        kv_caching (bool): If we are using KV-caching.
        model_sram (bool): If we are using SRAM in the model.
        dram_bandwidth (float): Bandwidth of the DRAM in GB/s.
        dram_active_power (float): Active power of the DRAM in W.

    Returns:
        (float, float): Energy consumed by DRAM in nJ, Energy consumed by the MHA comp in nJ.
    """
    #? model_sram, num_heads
    # if model_sram:
    #     return 0, energy_table("MHA", vector_size, seq_length)
    if kv_flag and not prefill: # only one token
        _mha_energy = energy_table("MHA", vector_size, seq_length) / seq_length
        _kv_cache_energy = dram_active_power * kv_cache_latency(vector_size, seq_length, dram_bandwidth)
    else:
        _mha_energy = energy_table("MHA", vector_size, seq_length)
        _kv_cache_energy = 0
    tri = 0.5 if causal else 1.0
    _mha_energy = _mha_energy * tri
    # TODO: we add the go cache results here
    if go_flag and not prefill:
        _go_cache_energy = dram_active_power * \
            go_cache_latency(vector_size, seq_length, num_selects, num_experts, dram_bandwidth)
    else:
        _go_cache_energy = 0
    return _go_cache_energy, _kv_cache_energy, _mha_energy


def calculate_mvm_latency(
    decoding_id: int,
    default: int = 100,
    factor: float = 1.0,
    # prefill: bool = True,
    dataflow: list[int] = [],
) -> int:
    if dataflow != None and len(dataflow) > 0:
        return dataflow[decoding_id]
    if decoding_id == 0: # prefill
        return math.ceil(default * factor)
    return math.ceil(default)


class AcceleratorConfig:
    """
    tiles (int): How many 3D tiles the accelerator has.
    tiers (int): How many 3D tiers per tile does the accelerator have.
    tier_shape (tuple[int]): What is the shape of each tier. E.g. (512,256).
    """

    def __init__(
        self,
        tiles: int,
        tiers: int,
        tier_shape: tuple[int, int],
        num_digital_units: int = 4,
        num_mha_units: int = 1,
        mvm_latency: int = 100,
        mvm_energy: float = 10,
        lock_mha_unit_to_layer: bool = False,
        lock_dpu_unit_to_layer: bool = False,
        model_sram: bool = True,
        model_ott: bool = True,
        kv_caching: bool = False,
        kv_flag: bool = False,
        go_flag: bool = False,
        moe_num_selects: int = 4,
        moe_num_experts: int = 16,
        dram_bandwidth: float = 5.332,
        dram_active_power: float = 0.2467,
        dram_inactive_power: float = 0.1297,
        passive_power: float = 0.02412, # W, 5 DPU and 4 MHA
        dataflow = [], # [decoding steps, num_layers]
        mvm_factor: float = 1.0,
        comm_factor: float = 1.0,
    ):
        self.tiles = tiles
        self.tiers = tiers
        self.tier_shape = tier_shape
        self.num_digital_units = num_digital_units
        self.num_mha_units = num_mha_units
        self.lock_mha_unit_to_layer = lock_mha_unit_to_layer
        self.lock_dpu_unit_to_layer = lock_dpu_unit_to_layer
        self.model_sram = model_sram
        self.model_ott = model_ott
        self.kv_caching = kv_caching
        self.kv_flag = kv_flag
        self.go_flag = go_flag
        self.moe_num_selects = moe_num_selects
        self.moe_num_experts = moe_num_experts
        self.dram_bandwidth = dram_bandwidth  # GB/s
        self.dram_active_power = dram_active_power  # W
        self.dram_inactive_power = dram_inactive_power  # W

        self.mvm_latency: Callable[[int], int] = lambda decoding_id: calculate_mvm_latency(
            decoding_id, mvm_latency, mvm_factor, dataflow, 
        )  # ns
        self.mha_latency: Callable[[int, int, bool], int] = partial(
            calculate_mha_latency,
            kv_flag=kv_flag,
            go_flag=go_flag,
            num_selects=moe_num_selects,
            num_experts=moe_num_experts,
            model_sram=model_sram,
            dram_bandwidth=dram_bandwidth,
        )
        # Unit [ns] this is a function of the vector_size
        self.layer_norm_latency: Callable[[int], int] = lru_cache(maxsize=4)(
            lambda vector_size: latency_table("LayerNorm", vector_size, 1)
        )
        # Normally it's inside FMA blocks, we'll put a phantom 1 ns latency (or remove it if it messes scheduling)
        self.relu_latency: Callable[[int], int] = lru_cache(maxsize=4)(
            lambda vector_size: latency_table("ReLU", vector_size, 1)
        )
        # Unit [ns] this is a function of the vector size
        self.gelu_latency: Callable[[int], int] = lru_cache(maxsize=4)(
            lambda vector_size: latency_table("GeLU", vector_size, 1)
        )
        # Unit [ns] this is a function of the vector size
        self.add_latency: Callable[[int], int] = lru_cache(maxsize=4)(
            lambda vector_size: latency_table("Add", vector_size, 1)
        )
        # Unit [ns] this is a function of the vector size
        self.com_latency: Callable[[int], int] = partial(
            communication_latency,
            seq_length=1,
            dram_bandwidth=dram_bandwidth,
            factor=comm_factor,
        )
        # Time to sample from softmax dist. in decoding step
        self.multinomial_latency: Callable[[int], int] = lru_cache(maxsize=4)(
            lambda vector_size: 10
        )

        # Energy numbers (in nJ)
        self.mvm_energy: float = mvm_energy  # 0.1 W power, 100 ns integration
        self.mha_energy: Callable[[int, int, bool], float] = partial(
            calculate_mha_energy,
            kv_flag=kv_flag,
            go_flag=go_flag,
            num_selects=moe_num_selects,
            num_experts=moe_num_experts,
            model_sram=model_sram,
            dram_bandwidth=dram_bandwidth,
            dram_active_power=dram_active_power,
        )
        # Unit [nJ] this is a function of the vector_size
        self.layer_norm_energy: Callable[[int], float] = lru_cache(maxsize=4)(
            lambda vector_size: energy_table("LayerNorm", vector_size, 1)
        )

        # Unit [nJ] this is a function of the vector_size
        self.relu_energy: Callable[[int], float] = lru_cache(maxsize=4)(
            lambda vector_size: energy_table("ReLU", vector_size, 1)
        )

        # Unit [nJ] this is a function of the vector_size
        self.gelu_energy: Callable[[int], float] = lru_cache(maxsize=4)(
            lambda vector_size: energy_table("GeLU", vector_size, 1)
        )

        # Unit [nJ] this is a function of the vector_size
        self.add_energy: Callable[[int], float] = lru_cache(maxsize=4)(
            lambda vector_size: energy_table("Add", vector_size, 1)
        )

        # Unit [nJ] this is a function of the vector_size (assumes 8 bits to be transferred)
        self.com_energy: Callable[[int], float] = partial(
            communication_energy,
            seq_length=1,
            dram_bandwidth=dram_bandwidth,
            dram_active_power=dram_active_power,
            factor=comm_factor,
        )

        self.multinomial_energy: Callable[[int], int] = lru_cache(maxsize=4)(
            lambda vector_size: 0.0
        )

        # Passive power (in W), adapt to how much power your units use
        # self.passive_power: float = (num_mha_units + num_digital_units) * 1.0
        self.passive_power: float = passive_power
        if self.kv_caching:
            self.passive_power += self.dram_inactive_power


class Accelerator:
    def __init__(self, config: AcceleratorConfig, device: str):
        """
        Acclelerator class that holds list of Tiles. Each Tile holds
        list of Tiers.

        Args:
            config (AcceleratorConfig): Configuration for accelerator.
        """
        self.config: AcceleratorConfig = config
        # - Initialize the blocks of the accelerator here
        # tiles (this is a resource), tiers, DPUs (resource)
        self.tiles: list = [
            Tile(
                config.tiers,
                tier_shape=config.tier_shape,
                name=f"tile_{idx}",
                accelerator_config=config,
                device=device,
            )
            for idx in range(config.tiles)
        ]


class Tile:
    def __init__(
        self,
        n_tiers: int,
        tier_shape: tuple[int],
        name: str,
        accelerator_config: AcceleratorConfig,
        device: str,
    ):
        """
        Tile class used in the Accelerator.

        Args:
            n_tiers (int): Number of tiers per Tile.
            tier_shape (tuple[int]): Shape of each tier.
            name (str): Name of the tile.
            accelerator_config (AcceleratorConfig): Configuration of the accelerator.
            device: (str): Device to be used.
        """
        self.logger = get_logger("Tile")
        self.tiers: list = [Tier(tier_shape, device=device) for _ in range(n_tiers)]
        self.name = name
        self.accelerator_config = accelerator_config
        # Number of times I wanted to use the tile, but it was
        # used by another MVM. It is calculated during runtime.
        self.num_conflicts: int = 0
        # How much time have we spent operating the tile, to be
        # used to calculate the mapping efficiency
        self.active_time: int = 0
        self.current_op = None


@torch.fx.wrap
def tier_linear(token, weight, op_info):
    return torch.nn.functional.linear(token, weight.T)


class Tier:
    def __init__(self, tier_shape: tuple[int], device: str):
        """
        Tier class that is used to perform a single MVM.

        Args:
            tier_shape (tuple[int]): Shape of the tier. E.g. (512,512).
            latency (int): Latency for executing one MVM.
            device: (str): Device to be used.
        """
        self.n_rows, self.n_cols = tier_shape
        self.name = ""
        self.mapping = None
        self.used = False
        self.logger = get_logger("Tier")
        self.weight = torch.randn(tier_shape, device=device)
        self.traceable = True
        self.is_mapped = False

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, token: Tensor, op_info: dict = None):
        assert self.traceable or token.ndim == 1, "token must have only one dim"
        d_token = token.numel()
        token = torch.nn.functional.pad(token, (0, self.n_rows - d_token))
        return tier_linear(token, self.weight, op_info=op_info)

    def set_name(self, name):
        self.name = name
