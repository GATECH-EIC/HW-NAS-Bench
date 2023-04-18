# [HW-NAS-Bench: Hardware-Aware Neural Architecture Search Benchmark](https://openreview.net/pdf?id=_0kaDkv3dVf)

Accepted as a spotlight paper at ICLR 2021.

## Table of content
+ [File structure](#file-structure)
+ [Prerequisites](#prerequisites)
+ [Preparation and download](#preparation-and-download)
+ [How to use HW-NAS-Bench](#how-to-use-hw-nas-bench)
+ [Misc](#misc)
+ [Acknowledgment](#acknowledgment)
## File structure
```bash
.
├── hw_nas_bench_api # HW-NAS-Bench API
│   ├── fbnet_models # FBNet's space
│   └── nas_201_models # NAS-Bench-201's space
│       ├── cell_infers
│       ├── cell_searchs
│       ├── config_utils
│       ├── shape_infers
│       └── shape_searchs
└── nas_201_api # NAS-Bench-201 API
```
## Prerequisites
The code has the following dependencies:

> + python >= 3.6.10
> + pytorch >= 1.2.0
> + numpy >= 1.18.5

## Preparation and download

No addtional file needs to be downloaded, our [HW-NAS-Bench dataset](HW-NAS-Bench-v1_0.pickle) has been included in this repository.

[Optional] If you want to use NAS-Bench-201 to access information about the architectures' accuracy and loss, please follow the [NAS-Bench-201 guide](https://github.com/D-X-Y/NAS-Bench-201/tree/6275241dd8cc25d39fa9618e4b9fa3ac2eda6d10), and download the [NAS-Bench-201-v1_1-096897.pth](https://drive.google.com/open?id=16Y0UwGisiouVRxW-W5hEtbxmcHw_0hF_).

## How to use HW-NAS-Bench
More usage can be found in our [jupyter notebook example](example.ipynb)

1. Create an API instance from a file:
```python
from hw_nas_bench_api import HWNASBenchAPI as HWAPI
hw_api = HWAPI("HW-NAS-Bench-v1_0.pickle", search_space="nasbench201")
```

2. Show the real measured/estimated hardware-cost in different datasets:
```python
# Example to get all the hardware metrics in the No.0,1,2 architectures under NAS-Bench-201's Space
for idx in range(3):
    for dataset in ["cifar10", "cifar100", "ImageNet16-120"]:
        HW_metrics = hw_api.query_by_index(idx, dataset)
        print("The HW_metrics (type: {}) for No.{} @ {} under NAS-Bench-201: {}".format(type(HW_metrics),

```
Corresponding printed information:
```bash
===> Example to get all the hardware metrics in the No.0,1,2 architectures under NAS-Bench-201's Space
The HW_metrics (type: <class 'dict'>) for No.0 @ cifar10 under NAS-Bench-201: {'edgegpu_latency': 5.807418537139893, 'edgegpu_energy': 24.226614330768584, 'raspi4_latency': 10.481976820010459, 'edgetpu_latency': 0.9571811309997429, 'pixel3_latency': 3.6058499999999998, 'eyeriss_latency': 3.645620000000001, 'eyeriss_energy': 0.6872827644999999, 'fpga_latency': 2.57296, 'fpga_energy': 18.01072}
...
```

3. Show the real measured/estimated hardware-cost for a single architecture:
```python
# Example to get use the hardware metrics in the No.0 architectures in CIFAR-10 under NAS-Bench-201's Space
print("===> Example to get use the hardware metrics in the No.0 architectures in CIFAR-10 under NAS-Bench-201's Space")
HW_metrics = hw_api.query_by_index(0, "cifar10")
for k in HW_metrics:
    if "latency" in k:
        unit = "ms"
    else:
        unit = "mJ"
    print("{}: {} ({})".format(k, HW_metrics[k], unit))
```
Corresponding printed information:
```bash
===> Example to get use the hardware metrics in the No.0 architectures in CIFAR-10 under NAS-Bench-201's Space
edgegpu_latency: 5.807418537139893 (ms)
edgegpu_energy: 24.226614330768584 (mJ)
raspi4_latency: 10.481976820010459 (ms)
edgetpu_latency: 0.9571811309997429 (ms)
pixel3_latency: 3.6058499999999998 (ms)
eyeriss_latency: 3.645620000000001 (ms)
eyeriss_energy: 0.6872827644999999 (mJ)
fpga_latency: 2.57296 (ms)
fpga_energy: 18.01072 (mJ)
```
4. Create the network from api:
```python
# Create the network
config = hw_api.get_net_config(0, "cifar10")
print(config)
from hw_nas_bench_api.nas_201_models import get_cell_based_tiny_net
network = get_cell_based_tiny_net(config) # create the network from configurration
print(network) # show the structure of this architecture
```
Corresponding printed information:
```bash
{'name': 'infer.tiny', 'C': 16, 'N': 5, 'arch_str': '|avg_pool_3x3~0|+|nor_conv_1x1~0|skip_connect~1|+|nor_conv_1x1~0|skip_connect~1|skip_connect~2|', 'num_classes': 10}
TinyNetwork(
  TinyNetwork(C=16, N=5, L=17)
  (stem): Sequential(
    (0): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (cells): ModuleList(
    (0): InferCell(
      info :: nodes=4, inC=16, outC=16, [1<-(I0-L0) | 2<-(I0-L1,I1-L2) | 3<-(I0-L3,I1-L4,I2-L5)], |avg_pool_3x3~0|+|nor_conv_1x1~0|skip_connect~1|+|nor_conv_1x1~0|skip_connect~1|skip_connect~2|
      (layers): ModuleList(
        (0): POOLING(
          (op): AvgPool2d(kernel_size=3, stride=1, padding=1)
        )
        (1): ReLUConvBN(
...
```

## Example of measuring hardware-cost 

When measuring the hardware-cost, a template is shared among different devices to collect the hardware-cost, as shown below. Different devices will use their own collect() function. 

```python
def collect(arch_idx, dataset):
    ...

device_name = "DEVICE"
import numpy as np
for dataset in DATASET_LIST: # repeat on the targe dataset
    metric_list = []
    for arch_idx in ARCH_IDX_LIST: # repeat on the architectures in the space
        metric = collect(arch_idx, dataset)
        metric_list.appencd(metric)
    assert len(metric_list) == len(ARCH_IDX_LIST)
    # save to npy
    metric_npy = np.array(metric_list)
    np.save("measurements_logs/{}/{}.npy".format(device_name, dataset), metric_npy)
```

For example, when measuring the latency and energy in the EdgeGPU, we replace the collect() with the following function, and os.system() is used to do each experiments on a specific architecture on a specific dataset sperately to avoid the remaining process in the system that brings extra errors. And the [EdgeGPU_Benchmark.py](dev/proj_edgegpu/EdgeGPU_Benchmark.py) is contained in the [project folder](dev/proj_edgegpu/).

```python
import os
import pickle
def collect(arch_idx, dataset):
    # CMD to run for measure one architecture @ one dataset
    CMD_to_run = "python3 EdgeGPU_Benchmark.py \
                  --arch_idx {} \
                  --log_label assigned_tasks \
                  --num_repeats_item 50 \
                  --dataset {}".format(arch_idx, dataset)
    os.system(CMD_to_run)
    path = os.path.join("measurements_logs", "{}_arch_idx_{}_num_repeats_{}_label_{}.pkl".format(dataset, arch_idx, 50, assigned_tasks))

    with open(path, 'rb') as f:
        res = pickle.load(f)
    return [res["energy"], res["latency"]]
```

A complete project folder is [here](dev/proj_edgegpu/).

## Misc

Part of the devices used in HW-NAS-Bench:

![Part of the devices used in HW-NAS-Bench](devices.jpg?raw=true "Devices")

## Acknowledgment
> + The code is inspired by [NAS-Bench-201](https://github.com/D-X-Y/NAS-Bench-201/tree/6275241dd8cc25d39fa9618e4b9fa3ac2eda6d10).

## Citation
```
@inproceedings{
li2021hwnasbench,
title={{\{}HW{\}}-{\{}NAS{\}}-Bench: Hardware-Aware Neural Architecture Search Benchmark},
author={Chaojian Li and Zhongzhi Yu and Yonggan Fu and Yongan Zhang and Yang Zhao and Haoran You and Qixuan Yu and Yue Wang and Yingyan Lin},
booktitle={International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=_0kaDkv3dVf}
}
```
## License
Copyright (c) 2022 GaTech-EIC. All rights reserved.

Licensed under the [MIT](https://github.com/GATECH-EIC/HW-NAS-Bench/blob/master/LICENSE) license.
