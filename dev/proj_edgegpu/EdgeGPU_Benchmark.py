import time
exp_time_start = time.time()
import pickle
import argparse
import os
from models import get_cell_based_tiny_net
import torch
import jetsonTX2Power as tx2pwr
import numpy as np
from torch2trt import torch2trt

parser = argparse.ArgumentParser(description='Reads the Jetson TX2 voltage and current sensors')
parser.add_argument('--arch_idx', type=int, help='the index of the architecture')
parser.add_argument('--log_label', type=str, help='the special label for log name')
parser.add_argument('--num_repeats', type=int, default=50, help='how many images are considered here')
parser.add_argument('--dataset', type=str, default="cifar10", help='the dataset to run')
parser.add_argument('--warm_up_time', type=int, default=3, help='How many seconds to warm up')
parser.add_argument('--add_exp_time', type=bool, default=False, help='Whether to compute the total experiments time')
args = parser.parse_args()

arch_idx = args.arch_idx
dataset = args.dataset
num_repeats = args.num_repeats
warm_up_time = args.warm_up_time
log_label = args.log_label

# prepare logs path
if not os.path.exists("measurements_logs"):
    os.mkdir("measurements_logs")

# prepare energy monitor
monitor = tx2pwr.EnergyMonitor()

# loading arch config
with open(os.path.join("arch_configs", dataset, "config_{}.pkl".format(arch_idx)), 'rb') as f:
    config = pickle.load(f)

# prepare network and inputs
network = get_cell_based_tiny_net(config)
network = network.cuda().eval()
if "cifar" in dataset:
    inputs = torch.rand(1,3,32,32).cuda() # batch size = 1, image size = 32*32*3
else:
    inputs = torch.rand(1,3,16,16).cuda() # batch size = 1, image size = 16*16*3

# convert to tensorrt models

network_trt = torch2trt(network, [inputs])
# warm up run
warmup_start = time.time()
warm_up_duration = 0.0
while warm_up_duration < warm_up_time:
    network_trt(inputs)
    torch.cuda.synchronize() # wait for cuda to finish (cuda is asynchronous!)
    warm_up_duration = time.time() - warmup_start

# Energy (mW) and Latency (s) Measurements
monitor.reset_energy_data_list()
monitor.start()
time_start = time.time()
for _ in range(num_repeats):
    network_trt(inputs)
    torch.cuda.synchronize() # wait for cuda to finish (cuda is asynchronous!)
time_end = time.time()

# Save per image energy (mJ) and latency (ms)
total_time = time_end-time_start # in s
avg_power = np.mean(np.array(monitor.energy_data_list)) # in mW

per_image_time = total_time*1000.0/num_repeats # in ms
per_image_energy = total_time/num_repeats*avg_power # in mJ

log_path = os.path.join("measurements_logs", "{}_arch_idx_{}_num_repeats_{}_label_{}.pkl".format(dataset, arch_idx, num_repeats, log_label))
exp_time_end = time.time()
exp_time = exp_time_end - exp_time_start


if args.add_exp_time:
    with open(log_path, 'wb') as f:
        pickle.dump({"latency":float(per_image_time), "energy":float(per_image_energy), "exp_time":exp_time}, f, pickle.HIGHEST_PROTOCOL)
    print(log_path, per_image_time, per_image_energy, exp_time)
else:
    with open(log_path, 'wb') as f:
        pickle.dump({"latency":float(per_image_time), "energy":float(per_image_energy)}, f, pickle.HIGHEST_PROTOCOL)
    print(log_path, per_image_time, per_image_energy)