# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Testing topi conv2d operator for VTA"""
import argparse
import json
import os

import pytest
import numpy as np
from collections import namedtuple

import tvm
from tvm import te
from tvm import relay
from tvm import autotvm, rpc
from tvm.contrib import utils
from tvm.contrib.pickle_memoize import memoize
from tvm import topi
import tvm.topi.testing
import vta
from vta import program_fpga, reconfig_runtime
import vta.testing
from vta.testing import simulator
import subprocess
import re
import os

sample_re = re.compile(".*write bytes = ([\d]+)\s+read bytes = ([\d]+)\s+ write b/w = ([\d\.]+)\s+read b/w = ([\d\.]+).*")
overall_re = re.compile(".*total write bytes = ([\d]+) and total read bytes = ([\d]+)")
Workload = namedtuple(
    "Conv2DWorkload",
    [
        "batch",
        "height",
        "width",
        "in_filter",
        "out_filter",
        "hkernel",
        "wkernel",
        "hpad",
        "wpad",
        "hstride",
        "wstride",
    ],
)

# Get batch info from env
env = vta.get_env()

# ResNet18 workloads
# resnet_wkls = [
#     # Workloads of resnet18 on imagenet
#     #('resnet-18.C1',  Workload(env.BATCH, 224, 224, 3,   64,  7, 7, 3, 3, 2, 2)),
#     ("resnet-18.C2", Workload(env.BATCH, 56, 56, 64, 64, 3, 3, 1, 1, 1, 1)),
#     ("resnet-18.C3", Workload(env.BATCH, 56, 56, 64, 128, 3, 3, 1, 1, 2, 2)),
#     ("resnet-18.C4", Workload(env.BATCH, 56, 56, 64, 128, 1, 1, 0, 0, 2, 2)),
#     ("resnet-18.C5", Workload(env.BATCH, 28, 28, 128, 128, 3, 3, 1, 1, 1, 1)),
#     ("resnet-18.C6", Workload(env.BATCH, 28, 28, 128, 256, 3, 3, 1, 1, 2, 2)),
#     ("resnet-18.C7", Workload(env.BATCH, 28, 28, 128, 256, 1, 1, 0, 0, 2, 2)),
#     ("resnet-18.C8", Workload(env.BATCH, 14, 14, 256, 256, 3, 3, 1, 1, 1, 1)),
#     ("resnet-18.C9", Workload(env.BATCH, 14, 14, 256, 512, 3, 3, 1, 1, 2, 2)),
#     ("resnet-18.C10", Workload(env.BATCH, 14, 14, 256, 512, 1, 1, 0, 0, 2, 2)),
#     ("resnet-18.C11", Workload(env.BATCH, 7, 7, 512, 512, 3, 3, 1, 1, 1, 1)),
# ]

#resnet_wkls = [
    # Workloads of resnet18 on imagenet
    #('resnet-18.C1',  Workload(env.BATCH, 224, 224, 3,   64,  7, 7, 3, 3, 2, 2)),
    #("resnet-18.C2", Workload(env.BATCH, 112, 112, 64, 128, 3, 3, 1, 1, 1, 1)),
    # ("resnet-18.C2", Workload(env.BATCH, 56, 56, 64, 64, 3, 3, 1, 1, 1, 1)),
    # ("resnet-18.C2", Workload(env.BATCH, 28, 28, 64, 64, 3, 3, 1, 1, 1, 1)),
    # ("resnet-18.C2", Workload(env.BATCH, 14, 14, 64, 64, 3, 3, 1, 1, 1, 1)),
    # ("resnet-18.C11", Workload(env.BATCH, 7, 7, 64, 64, 3, 3, 1, 1, 1, 1)),
#]

resnet_wkls = [
    # ("workload_0", Workload(env.BATCH, 208, 208, 16, 32, 3, 3, 1, 1, 1, 1)),
    # ("workload_1", Workload(env.BATCH, 104, 104, 32, 64, 3, 3, 1, 1, 1, 1)),
    # ("workload_2", Workload(env.BATCH, 26, 26, 128, 256, 3, 3, 1, 1, 1, 1)),
    # ("workload_3", Workload(env.BATCH, 13, 13, 256, 512, 3, 3, 1, 1, 1, 1)),
    # ("workload_4", Workload(env.BATCH, 13, 13, 512, 1024, 3, 3, 1, 1, 1, 1)),
    # ("workload_5", Workload(env.BATCH, 52, 52, 64, 128, 3, 3, 1, 1, 1, 1)),
    # ("workload_6", Workload(env.BATCH, 13, 13, 256, 128, 1, 1, 0, 0, 1, 1)),
    # ("workload_7", Workload(env.BATCH, 13, 13, 1024, 256, 1, 1, 0, 0, 1, 1)),
    # ("workload_8", Workload(env.BATCH, 26, 26, 384, 256, 3, 3, 1, 1, 1, 1)),
    # ("workload_9", Workload(env.BATCH, 13, 13, 512, 256, 1, 1, 0, 0, 1, 1)),
    # ("workload_10", Workload(env.BATCH, 26, 26, 256, 256, 1, 1, 0, 0, 1, 1)),
    # ("workload_11", Workload(env.BATCH, 14, 14, 256, 512, 3, 3, 1, 1, 2, 2)),
    # ("workload_12", Workload(env.BATCH, 7, 7, 512, 512, 3, 3, 1, 1, 1, 1)),
    # ("workload_13", Workload(env.BATCH, 28, 28, 128, 256, 3, 3, 1, 1, 2, 2)),
    # ("workload_14", Workload(env.BATCH, 14, 14, 256, 256, 3, 3, 1, 1, 1, 1)),
    # ("workload_15", Workload(env.BATCH, 56, 56, 64, 128, 3, 3, 1, 1, 2, 2)),
    ("workload_16", Workload(env.BATCH, 28, 28, 128, 128, 3, 3, 1, 1, 1, 1)),
    # ("workload_17", Workload(env.BATCH, 56, 56, 64, 64, 3, 3, 1, 1, 1, 1)),
    # ("workload_18", Workload(env.BATCH, 56, 56, 64, 128, 1, 1, 0, 0, 2, 2)),
    # ("workload_19", Workload(env.BATCH, 28, 28, 128, 256, 1, 1, 0, 0, 2, 2)),
    # ("workload_20", Workload(env.BATCH, 14, 14, 256, 512, 1, 1, 0, 0, 2, 2)),
    # ("workload_21", Workload(env.BATCH, 14, 14, 1024, 2048, 1, 1, 0, 0, 2, 2)),
    # ("workload_22", Workload(env.BATCH, 28, 28, 512, 1024, 1, 1, 0, 0, 2, 2)),
    # ("workload_23", Workload(env.BATCH, 56, 56, 256, 512, 1, 1, 0, 0, 2, 2)),
    # ("workload_24", Workload(env.BATCH, 56, 56, 64, 64, 1, 1, 0, 0, 1, 1)),
    # ("workload_25", Workload(env.BATCH, 56, 56, 256, 64, 1, 1, 0, 0, 1, 1)),
    # ("workload_26", Workload(env.BATCH, 56, 56, 64, 256, 1, 1, 0, 0, 1, 1)),
    # ("workload_27", Workload(env.BATCH, 56, 56, 256, 128, 1, 1, 0, 0, 1, 1)),
    # ("workload_28", Workload(env.BATCH, 56, 56, 128, 128, 3, 3, 1, 1, 2, 2)),
    # ("workload_29", Workload(env.BATCH, 28, 28, 512, 128, 1, 1, 0, 0, 1, 1)),
    # ("workload_30", Workload(env.BATCH, 28, 28, 128, 512, 1, 1, 0, 0, 1, 1)),
    # ("workload_31", Workload(env.BATCH, 28, 28, 512, 256, 1, 1, 0, 0, 1, 1)),
    # ("workload_32", Workload(env.BATCH, 28, 28, 256, 256, 3, 3, 1, 1, 2, 2)),
    # ("workload_33", Workload(env.BATCH, 14, 14, 1024, 256, 1, 1, 0, 0, 1, 1)),
    # ("workload_34", Workload(env.BATCH, 14, 14, 256, 1024, 1, 1, 0, 0, 1, 1)),
    # ("workload_35", Workload(env.BATCH, 14, 14, 1024, 512, 1, 1, 0, 0, 1, 1)),
    # ("workload_36", Workload(env.BATCH, 14, 14, 512, 512, 3, 3, 1, 1, 2, 2)),
    # ("workload_37", Workload(env.BATCH, 7, 7, 2048, 512, 1, 1, 0, 0, 1, 1)),
    # ("workload_38", Workload(env.BATCH, 7, 7, 512, 2048, 1, 1, 0, 0, 1, 1)),
    # ("workload_39", Workload(env.BATCH, 27, 27, 64, 192, 5, 5, 2, 2, 1, 1)),
    # ("workload_40", Workload(env.BATCH, 13, 13, 192, 384, 3, 3, 1, 1, 1, 1)),
    # ("workload_41", Workload(env.BATCH, 13, 13, 384, 256, 3, 3, 1, 1, 1, 1)),
    # ("workload_42", Workload(env.BATCH, 13, 13, 256, 256, 3, 3, 1, 1, 1, 1)),
    # ("workload_43", Workload(env.BATCH, 112, 112, 64, 128, 3, 3, 1, 1, 1, 1)),
    # ("workload_44", Workload(env.BATCH, 112, 112, 128, 128, 3, 3, 1, 1, 1, 1)),
    # ("workload_45", Workload(env.BATCH, 56, 56, 128, 256, 3, 3, 1, 1, 1, 1)),
    # ("workload_46", Workload(env.BATCH, 56, 56, 256, 256, 3, 3, 1, 1, 1, 1)),
    # ("workload_47", Workload(env.BATCH, 28, 28, 256, 512, 3, 3, 1, 1, 1, 1)),
    # ("workload_48", Workload(env.BATCH, 28, 28, 512, 512, 3, 3, 1, 1, 1, 1)),
    # ("workload_49", Workload(env.BATCH, 14, 14, 512, 512, 3, 3, 1, 1, 1, 1)),

]
# FIXME: we need a custom clip operator to circumvent a pattern detection limitation
@tvm.te.tag_scope(tag=topi.tag.ELEMWISE)
def my_clip(x, a_min, a_max):
    """Unlike topi's current clip, put min and max into two stages."""
    const_min = tvm.tir.const(a_min, x.dtype)
    const_max = tvm.tir.const(a_max, x.dtype)
    x = te.compute(x.shape, lambda *i: tvm.te.min(x(*i), const_max), name="clipA")
    x = te.compute(x.shape, lambda *i: tvm.te.max(x(*i), const_min), name="clipB")
    return x


def run_conv2d(env, remote, wl, target, check_correctness=True, print_ir=False, samples=4, log_file='logs/log.json'):

    # Workload assertions
    assert wl.hpad == wl.wpad

    if not os.path.exists(log_file):
        with open(log_file, 'w+') as myfile:
            json.dump({"workloads": []}, myfile, indent=4)

    # Perform packing only if we are targeting the accelerator
    if "arm_cpu" in target.keys:
        data_pack = False
        layout = "NCHW"
        conv2d_fcompute = topi.arm_cpu.conv2d_nchw_spatial_pack
        conv2d_fschedule = topi.arm_cpu.schedule_conv2d_nchw_spatial_pack
    elif "vta" in target.keys:
        data_pack = True
        layout = "NCHW%dn%dc" % (env.BATCH, env.BLOCK_IN)
        conv2d_fcompute = vta.top.conv2d_packed
        conv2d_fschedule = vta.top.schedule_conv2d_packed

    # Derive shapes depending upon packing
    a_shape = (wl.batch, wl.in_filter, wl.height, wl.width)
    w_shape = (wl.out_filter, wl.in_filter, wl.hkernel, wl.wkernel)
    b_shape = (wl.batch, wl.out_filter, 1, 1)

    if data_pack:
        data_shape = (
            wl.batch // env.BATCH,
            wl.in_filter // env.BLOCK_IN,
            wl.height,
            wl.width,
            env.BATCH,
            env.BLOCK_IN,
        )
        kernel_shape = (
            wl.out_filter // env.BLOCK_OUT,
            wl.in_filter // env.BLOCK_IN,
            wl.hkernel,
            wl.wkernel,
            env.BLOCK_OUT,
            env.BLOCK_IN,
        )
        bias_shape = (
            wl.batch // env.BATCH,
            wl.out_filter // env.BLOCK_OUT,
            1,
            1,
            env.BATCH,
            env.BLOCK_OUT,
        )
    else:
        data_shape = a_shape
        kernel_shape = w_shape
        bias_shape = b_shape
    data = te.placeholder(data_shape, name="data", dtype=env.inp_dtype)
    kernel = te.placeholder(kernel_shape, name="kernel", dtype=env.wgt_dtype)
    bias = te.placeholder(bias_shape, name="bias", dtype=env.acc_dtype)
    padding = relay.nn.get_pad_tuple2d((wl.hpad, wl.wpad))

    # Define base computation schedule
    with target:

        if data_pack:
            res = conv2d_fcompute(
                data, kernel, (wl.hstride, wl.wstride), padding, (1, 1), layout, env.acc_dtype
            )
        else:
            res = conv2d_fcompute(
                data, kernel, (wl.hstride, wl.wstride), padding, (1, 1), env.acc_dtype
            )
        res = topi.right_shift(res, 8)
        res = topi.add(res, bias)
        res = my_clip(res, 0, (1 << env.OUT_WIDTH - 1) - 1)
        res = topi.cast(res, env.out_dtype)

        # Derive base schedule
        s = conv2d_fschedule([res])
        if print_ir:
            print(vta.lower(s, [data, kernel, bias, res], simple_mode=True))

    # Derive number of ops
    fout_height = (wl.height + 2 * wl.hpad - wl.hkernel) // wl.hstride + 1
    fout_width = (wl.width + 2 * wl.wpad - wl.wkernel) // wl.wstride + 1
    num_ops = (
        2
        * wl.batch
        * fout_height
        * fout_width
        * wl.hkernel
        * wl.wkernel
        * wl.out_filter
        * wl.in_filter
    )

    # @memoize("vta.tests.test_benchmark_topi.conv2d.verify_nhwc")
    def get_ref_data():
        # derive min max for act, wgt, and bias types (max non inclusive)
        a_min, a_max = 0 - (1 << (env.INP_WIDTH - 1)), (1 << (env.INP_WIDTH - 1))
        w_min, w_max = 0 - (1 << (env.WGT_WIDTH - 1)), (1 << (env.WGT_WIDTH - 1))
        b_min, b_max = 0 - 1 << (env.INP_WIDTH + env.WGT_WIDTH - 2), 1 << (
            env.INP_WIDTH + env.WGT_WIDTH - 2
        )
        a_np = np.random.randint(a_min, a_max, size=a_shape).astype(data.dtype)
        w_np = np.random.randint(w_min, w_max, size=w_shape).astype(kernel.dtype)
        b_np = np.random.randint(b_min, b_max, size=b_shape).astype(env.acc_dtype)
        r_np = tvm.topi.testing.conv2d_nchw_python(
            a_np.astype(env.acc_dtype),
            w_np.astype(env.acc_dtype),
            (wl.hstride, wl.wstride),
            wl.hpad,
        ).astype(env.acc_dtype)
        return a_np, w_np, b_np, r_np

    # Data in original format
    data_np, kernel_np, bias_np, res_ref = get_ref_data()
    if data_pack:
        data_np = data_np.reshape(
            wl.batch // env.BATCH,
            env.BATCH,
            wl.in_filter // env.BLOCK_IN,
            env.BLOCK_IN,
            wl.height,
            wl.width,
        ).transpose((0, 2, 4, 5, 1, 3))
        kernel_np = kernel_np.reshape(
            wl.out_filter // env.BLOCK_OUT,
            env.BLOCK_OUT,
            wl.in_filter // env.BLOCK_IN,
            env.BLOCK_IN,
            wl.hkernel,
            wl.wkernel,
        ).transpose((0, 2, 4, 5, 1, 3))
        bias_np = bias_np.reshape(
            wl.batch // env.BATCH, wl.out_filter // env.BLOCK_OUT, 1, 1, env.BATCH, env.BLOCK_OUT
        )
    # Build
    if "vta" in target.keys:
        mod = vta.build(
            s,
            [data, kernel, bias, res],
            target=tvm.target.Target(target, host=env.target_host),
            name="conv2d",
        )
    else:
        mod = tvm.build(
            s,
            [data, kernel, bias, res],
            target=tvm.target.Target(target, host=env.target_host),
            name="conv2d",
        )
    temp = utils.tempdir()
    mod.save(temp.relpath("conv2d.o"))
    remote.upload(temp.relpath("conv2d.o"))
    f = remote.load_module("conv2d.o")
    dev = remote.device(str(target))

    res_np = np.zeros(topi.utils.get_const_tuple(res.shape)).astype(res.dtype)
    data_arr = tvm.nd.array(data_np, dev)
    kernel_arr = tvm.nd.array(kernel_np, dev)
    bias_arr = tvm.nd.array(bias_np, dev)
    res_arr = tvm.nd.array(res_np, dev)
    #time_f = f.time_evaluator("conv2d", dev, number=samples)

    result_dict = {0: {}, 1: {}, 2: {}, 3: {}, 4: {}, 5: {}}
    workload_dict = {"height": wl.height, "width": wl.width, "in_filter": wl.in_filter, "out_filter": wl.out_filter,
                     "hkernel": wl.hkernel, "wkernel": wl.wkernel, "hpad": wl.hpad, "wpad": wl.wpad,
                     "hstride": wl.hstride, "wstride": wl.wstride, "results": {},
                     "workload_str":'({},{},{},{},{},{},{},{},{},{})'.format(wl.height, wl.width, wl.in_filter,
                                                                             wl.out_filter, wl.hkernel, wl.wkernel,
                                                                             wl.hpad, wl.wpad, wl.hstride, wl.wstride)}
    #with open("/home/srchand/Desktop/research/TVM/tvm/vta/sri_trial/logs/conv_profiling_results.txt", 'a') as myfile:
        #myfile.write("\n")
        #myfile.write(str(wl))
    for slot in range(6):
        per_slot_dict = {"samples": [], "overall": {"write_bytes": 0, "read_bytes": 0}}
        #myfile.write("\nSlot {} ".format(slot))

        print("starting polling subprocess")
        proc = subprocess.Popen(["sshpass", "-p", "Srivat95", "ssh", "-t", "xilinx@192.168.2.99", "sudo", "python3",
                             "/home/xilinx/tvm/vta/python/vta/poll_apm.py", "--slot", str(slot)], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
        count = 0
        for i in range(200000000):
            count += 1
        f(data_arr, kernel_arr, bias_arr, res_arr)

        while True:
            line = proc.stdout.readline()
            if not line:
                break
            line = line.decode("utf-8")
            match = re.search(sample_re, line)
            if match:
                #print("write {} read {} write bw {} read bw {}".format(match.group(1),match.group(2),match.group(3),match.group(4)))
                if match.group(1) != '0' or match.group(2) != '0':
                    #result_dict[slot].append([match.group(1),match.group(2),match.group(3),match.group(4)])
                    per_sample_dict = {"write_bytes": int(match.group(1)), "read_bytes": int(match.group(2)),
                                       "write_bw": float(match.group(3)), "read_bw": float(match.group(4))}
                    per_slot_dict["samples"].append(per_sample_dict)
            else:
                match = re.search(overall_re,line)
                if match:
                    per_slot_dict["overall"]["write_bytes"] = int(match.group(1))
                    per_slot_dict["overall"]["read_bytes"] = int(match.group(2))

                    #result_dict[slot].append([match.group(1), match.group(2)])

        result_dict[slot] = per_slot_dict
        #myfile.write(str(result_dict[slot]))

        print("Ran convolution successfully!!")
    workload_dict["results"] = result_dict
    with open(log_file, 'r+') as myfile:
        file_data = json.load(myfile)
        file_data["workloads"].append(workload_dict)
        myfile.seek(0)
        json.dump(file_data, myfile, indent=4)

def test_conv2d(device, log_file = "logs/log.json"):
    device_host = os.environ.get("VTA_RPC_HOST", "192.168.2.99")
    device_port = os.environ.get("VTA_RPC_PORT", "9091")
    remote = rpc.connect(device_host, int(device_port))
    if device == "vta":
        target = env.target
        if env.TARGET not in ["sim", "tsim", "intelfocl"]:
            assert tvm.runtime.enabled("rpc")
            program_fpga(remote, bitstream="/home/srchand/Desktop/overlays/vta/vta_apm_6slots.bit")
            reconfig_runtime(remote)
    elif device == "arm_cpu":
        target = env.target_vta_cpu
    with autotvm.tophub.context(target):  # load pre-tuned schedule parameters
        for _, wl in resnet_wkls:
            print(wl)
            run_conv2d(env, remote, wl, target, log_file=log_file)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AXI Performance Monitor Convolution Benchmark')
    parser.add_argument('--log_file', type=str, default="logs/log.json",
                        help='output log file path')

    args = parser.parse_args()
    #test_conv2d(device="arm_cpu")
    test_conv2d(device="vta", log_file = args.log_file)


