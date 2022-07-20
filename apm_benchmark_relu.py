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
import time
from tvm.contrib import graph_executor
import pytest
import numpy as np
from collections import namedtuple
from vta.top import graph_pack

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
import torch
from torch import nn

schedule_log_files = ['logs/tuning_logs/vta.regnet_x_400mf.log',
                      'logs/tuning_logs/vta.googlenet.log',
                      'logs/tuning_logs/vta.squeezenet1.1.log',
                      'logs/tuning_logs/vta.mobilenet_v2.log',
                      'logs/tuning_logs/vta.regnet_x_800mf.log',
                      'logs/tuning_logs/vta.inceptionv3.log',
                      'logs/tuning_logs/vta.vgg11.log',
                      'logs/tuning_logs/vta.wide_resnet50_2.log',
                      'logs/tuning_logs/vta.regnet_x_1_6gf.log',
                      'logs/tuning_logs/vta.regnet_x_3_2gf.log',
                      'logs/tuning_logs/vta.vgg16.log',
                      'logs/tuning_logs/vta.mnasnet_0_5.log',
                      'logs/tuning_logs/vta.resnext50_32x4d.log']

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

# resnet_wkls = [
#     # ("workload_0", Workload(env.BATCH, 208, 208, 16, 32, 3, 3, 1, 1, 1, 1)),
#     # ("workload_1", Workload(env.BATCH, 104, 104, 32, 64, 3, 3, 1, 1, 1, 1)),
#     # ("workload_2", Workload(env.BATCH, 26, 26, 128, 256, 3, 3, 1, 1, 1, 1)),
#     # ("workload_3", Workload(env.BATCH, 13, 13, 256, 512, 3, 3, 1, 1, 1, 1)),
#     # ("workload_4", Workload(env.BATCH, 13, 13, 512, 1024, 3, 3, 1, 1, 1, 1)),
#     # ("workload_5", Workload(env.BATCH, 52, 52, 64, 128, 3, 3, 1, 1, 1, 1)),
#     # ("workload_6", Workload(env.BATCH, 13, 13, 256, 128, 1, 1, 0, 0, 1, 1)),
#     # ("workload_7", Workload(env.BATCH, 13, 13, 1024, 256, 1, 1, 0, 0, 1, 1)),
#     # ("workload_8", Workload(env.BATCH, 26, 26, 384, 256, 3, 3, 1, 1, 1, 1)),
#     # ("workload_9", Workload(env.BATCH, 13, 13, 512, 256, 1, 1, 0, 0, 1, 1)),
#     # ("workload_10", Workload(env.BATCH, 26, 26, 256, 256, 1, 1, 0, 0, 1, 1)),
#     ("workload_11", Workload(env.BATCH, 14, 14, 256, 512, 3, 3, 1, 1, 2, 2)),
#     # ("workload_12", Workload(env.BATCH, 7, 7, 512, 512, 3, 3, 1, 1, 1, 1)),
#     # ("workload_13", Workload(env.BATCH, 28, 28, 128, 256, 3, 3, 1, 1, 2, 2)),
#     # ("workload_14", Workload(env.BATCH, 14, 14, 256, 256, 3, 3, 1, 1, 1, 1)),
#     # ("workload_15", Workload(env.BATCH, 56, 56, 64, 128, 3, 3, 1, 1, 2, 2)),
#     # ("workload_16", Workload(env.BATCH, 28, 28, 128, 128, 3, 3, 1, 1, 1, 1)),
#     # ("workload_17", Workload(env.BATCH, 56, 56, 64, 64, 3, 3, 1, 1, 1, 1)),
#     # ("workload_18", Workload(env.BATCH, 56, 56, 64, 128, 1, 1, 0, 0, 2, 2)),
#     # ("workload_19", Workload(env.BATCH, 28, 28, 128, 256, 1, 1, 0, 0, 2, 2)),
#     # ("workload_20", Workload(env.BATCH, 14, 14, 256, 512, 1, 1, 0, 0, 2, 2)),
#     # ("workload_21", Workload(env.BATCH, 14, 14, 1024, 2048, 1, 1, 0, 0, 2, 2)),
#     # ("workload_22", Workload(env.BATCH, 28, 28, 512, 1024, 1, 1, 0, 0, 2, 2)),
#     # ("workload_23", Workload(env.BATCH, 56, 56, 256, 512, 1, 1, 0, 0, 2, 2)),
#     # ("workload_24", Workload(env.BATCH, 56, 56, 64, 64, 1, 1, 0, 0, 1, 1)),
#     # ("workload_25", Workload(env.BATCH, 56, 56, 256, 64, 1, 1, 0, 0, 1, 1)),
#     # ("workload_26", Workload(env.BATCH, 56, 56, 64, 256, 1, 1, 0, 0, 1, 1)),
#     # ("workload_27", Workload(env.BATCH, 56, 56, 256, 128, 1, 1, 0, 0, 1, 1)),
#     # ("workload_28", Workload(env.BATCH, 56, 56, 128, 128, 3, 3, 1, 1, 2, 2)),
#     # ("workload_29", Workload(env.BATCH, 28, 28, 512, 128, 1, 1, 0, 0, 1, 1)),
#     # ("workload_30", Workload(env.BATCH, 28, 28, 128, 512, 1, 1, 0, 0, 1, 1)),
#     # ("workload_31", Workload(env.BATCH, 28, 28, 512, 256, 1, 1, 0, 0, 1, 1)),
#     # ("workload_32", Workload(env.BATCH, 28, 28, 256, 256, 3, 3, 1, 1, 2, 2)),
#     # ("workload_33", Workload(env.BATCH, 14, 14, 1024, 256, 1, 1, 0, 0, 1, 1)),
#     # ("workload_34", Workload(env.BATCH, 14, 14, 256, 1024, 1, 1, 0, 0, 1, 1)),
#     # ("workload_35", Workload(env.BATCH, 14, 14, 1024, 512, 1, 1, 0, 0, 1, 1)),
#     # ("workload_36", Workload(env.BATCH, 14, 14, 512, 512, 3, 3, 1, 1, 2, 2)),
#     # ("workload_37", Workload(env.BATCH, 7, 7, 2048, 512, 1, 1, 0, 0, 1, 1)),
#     # ("workload_38", Workload(env.BATCH, 7, 7, 512, 2048, 1, 1, 0, 0, 1, 1)),
#     # ("workload_39", Workload(env.BATCH, 27, 27, 64, 192, 5, 5, 2, 2, 1, 1)),
#     # ("workload_40", Workload(env.BATCH, 13, 13, 192, 384, 3, 3, 1, 1, 1, 1)),
#     # ("workload_41", Workload(env.BATCH, 13, 13, 384, 256, 3, 3, 1, 1, 1, 1)),
#     # ("workload_42", Workload(env.BATCH, 13, 13, 256, 256, 3, 3, 1, 1, 1, 1)),
#     # ("workload_43", Workload(env.BATCH, 112, 112, 64, 128, 3, 3, 1, 1, 1, 1)),
#     # ("workload_44", Workload(env.BATCH, 112, 112, 128, 128, 3, 3, 1, 1, 1, 1)),
#     # ("workload_45", Workload(env.BATCH, 56, 56, 128, 256, 3, 3, 1, 1, 1, 1)),
#     # ("workload_46", Workload(env.BATCH, 56, 56, 256, 256, 3, 3, 1, 1, 1, 1)),
#     # ("workload_47", Workload(env.BATCH, 28, 28, 256, 512, 3, 3, 1, 1, 1, 1)),
#     # ("workload_48", Workload(env.BATCH, 28, 28, 512, 512, 3, 3, 1, 1, 1, 1)),
#     # ("workload_49", Workload(env.BATCH, 14, 14, 512, 512, 3, 3, 1, 1, 1, 1)),
# ]

resnet_wkls = [
    # ('workloads_0', Workload(1, 56, 56, 64, 64, 1, 1, 0, 0, 1, 1)),
    # ('workloads_1', Workload(1, 56, 56, 64, 192, 3, 3, 1, 1, 1, 1)),
    # ('workloads_2', Workload(1, 28, 28, 192, 64, 1, 1, 0, 0, 1, 1)),
    # ('workloads_3', Workload(1, 28, 28, 192, 96, 1, 1, 0, 0, 1, 1)),
    # ('workloads_4', Workload(1, 28, 28, 96, 128, 3, 3, 1, 1, 1, 1)),
    # ('workloads_5', Workload(1, 28, 28, 192, 16, 1, 1, 0, 0, 1, 1)),
    # ('workloads_6', Workload(1, 28, 28, 16, 32, 3, 3, 1, 1, 1, 1)),
    # ('workloads_7', Workload(1, 28, 28, 192, 32, 1, 1, 0, 0, 1, 1)),
    # ('workloads_8', Workload(1, 28, 28, 256, 128, 1, 1, 0, 0, 1, 1)),
    # ('workloads_9', Workload(1, 28, 28, 128, 192, 3, 3, 1, 1, 1, 1)),
    # ('workloads_10', Workload(1, 28, 28, 256, 32, 1, 1, 0, 0, 1, 1)),
    # ('workloads_11', Workload(1, 28, 28, 32, 96, 3, 3, 1, 1, 1, 1)),
    # ('workloads_12', Workload(1, 28, 28, 256, 64, 1, 1, 0, 0, 1, 1)),
    # ('workloads_13', Workload(1, 28, 28, 480, 192, 1, 1, 0, 0, 1, 1)),
    # ('workloads_14', Workload(1, 14, 14, 480, 96, 1, 1, 0, 0, 1, 1)),
    # ('workloads_15', Workload(1, 14, 14, 96, 208, 3, 3, 1, 1, 1, 1)),
    # ('workloads_16', Workload(1, 14, 14, 480, 16, 1, 1, 0, 0, 1, 1)),
    # ('workloads_17', Workload(1, 14, 14, 16, 48, 3, 3, 1, 1, 1, 1)),
    # ('workloads_18', Workload(1, 14, 14, 480, 64, 1, 1, 0, 0, 1, 1)),
    # ('workloads_19', Workload(1, 14, 14, 512, 160, 1, 1, 0, 0, 1, 1)),
    # ('workloads_20', Workload(1, 14, 14, 512, 112, 1, 1, 0, 0, 1, 1)),
    # ('workloads_21', Workload(1, 14, 14, 112, 224, 3, 3, 1, 1, 1, 1)),
    # ('workloads_22', Workload(1, 14, 14, 512, 64, 1, 1, 0, 0, 1, 1)),
    # ('workloads_23', Workload(1, 14, 14, 512, 128, 1, 1, 0, 0, 1, 1)),
    # ('workloads_24', Workload(1, 14, 14, 128, 256, 3, 3, 1, 1, 1, 1)),
    # ('workloads_25', Workload(1, 14, 14, 512, 144, 1, 1, 0, 0, 1, 1)),
    # ('workloads_26', Workload(1, 14, 14, 144, 288, 3, 3, 1, 1, 1, 1)),
    # ('workloads_27', Workload(1, 14, 14, 512, 32, 1, 1, 0, 0, 1, 1)),
    # ('workloads_28', Workload(1, 14, 14, 32, 64, 3, 3, 1, 1, 1, 1)),
    # ('workloads_29', Workload(1, 14, 14, 528, 256, 1, 1, 0, 0, 1, 1)),
    # ('workloads_30', Workload(1, 14, 14, 528, 160, 1, 1, 0, 0, 1, 1)),
    # ('workloads_31', Workload(1, 14, 14, 160, 320, 3, 3, 1, 1, 1, 1)),
    # ('workloads_32', Workload(1, 14, 14, 528, 32, 1, 1, 0, 0, 1, 1)),
    # ('workloads_33', Workload(1, 14, 14, 32, 128, 3, 3, 1, 1, 1, 1)),
    # ('workloads_34', Workload(1, 14, 14, 528, 128, 1, 1, 0, 0, 1, 1)),
    # ('workloads_35', Workload(1, 14, 14, 832, 256, 1, 1, 0, 0, 1, 1)),
    # ('workloads_36', Workload(1, 7, 7, 832, 160, 1, 1, 0, 0, 1, 1)),
    # ('workloads_37', Workload(1, 7, 7, 160, 320, 3, 3, 1, 1, 1, 1)),
    # ('workloads_38', Workload(1, 7, 7, 832, 32, 1, 1, 0, 0, 1, 1)),
    # ('workloads_39', Workload(1, 7, 7, 32, 128, 3, 3, 1, 1, 1, 1)),
    # ('workloads_40', Workload(1, 7, 7, 832, 128, 1, 1, 0, 0, 1, 1)),
    # ('workloads_41', Workload(1, 7, 7, 832, 384, 1, 1, 0, 0, 1, 1)),
    # ('workloads_42', Workload(1, 7, 7, 832, 192, 1, 1, 0, 0, 1, 1)),
    # ('workloads_43', Workload(1, 7, 7, 192, 384, 3, 3, 1, 1, 1, 1)),
    # ('workloads_44', Workload(1, 7, 7, 832, 48, 1, 1, 0, 0, 1, 1)),
    # ('workloads_45', Workload(1, 7, 7, 48, 128, 3, 3, 1, 1, 1, 1)),
    # ('workloads_46', Workload(1, 56, 56, 16, 16, 3, 3, 1, 1, 1, 1)),
    # ('workloads_47', Workload(1, 56, 56, 16, 48, 1, 1, 0, 0, 1, 1)),
    # ('workloads_48', Workload(1, 56, 56, 48, 48, 3, 3, 1, 1, 1, 1)),
    # ('workloads_49', Workload(1, 56, 56, 48, 16, 1, 1, 0, 0, 1, 1)),
    # ('workloads_50', Workload(1, 14, 14, 48, 48, 3, 3, 1, 1, 1, 1)),
    # ('workloads_51', Workload(1, 14, 14, 48, 16, 1, 1, 0, 0, 1, 1)),
    # ('workloads_52', Workload(1, 14, 14, 16, 48, 1, 1, 0, 0, 1, 1)),
    # ('workloads_53', Workload(1, 14, 14, 48, 48, 5, 5, 2, 2, 2, 2)),
    # ('workloads_54', Workload(1, 7, 7, 144, 144, 5, 5, 2, 2, 2, 2)),
    # ('workloads_55', Workload(1, 7, 7, 240, 240, 5, 5, 2, 2, 1, 1)),
    # ('workloads_56', Workload(1, 7, 7, 240, 240, 3, 3, 1, 1, 1, 1)),
    # ('workloads_57', Workload(1, 7, 7, 240, 48, 1, 1, 0, 0, 1, 1)),
    # ('workloads_58', Workload(1, 7, 7, 48, 288, 1, 1, 0, 0, 1, 1)),
    # ('workloads_59', Workload(1, 7, 7, 288, 288, 3, 3, 1, 1, 1, 1)),
    # ('workloads_60', Workload(1, 7, 7, 288, 48, 1, 1, 0, 0, 1, 1)),
    # ('workloads_61', Workload(1, 7, 7, 288, 288, 5, 5, 2, 2, 2, 2)),
    # ('workloads_62', Workload(1, 112, 112, 32, 32, 3, 3, 1, 1, 1, 1)),
    # ('workloads_63', Workload(1, 112, 112, 32, 16, 1, 1, 0, 0, 1, 1)),
    # ('workloads_64', Workload(1, 28, 28, 16, 96, 1, 1, 0, 0, 1, 1)),
    # ('workloads_65', Workload(1, 28, 28, 96, 96, 3, 3, 1, 1, 2, 2)),
    # ('workloads_66', Workload(1, 28, 28, 144, 144, 3, 3, 1, 1, 1, 1)),
    # ('workloads_67', Workload(1, 28, 28, 144, 144, 3, 3, 1, 1, 2, 2)),
    # ('workloads_68', Workload(1, 28, 28, 144, 32, 1, 1, 0, 0, 1, 1)),
    # ('workloads_69', Workload(1, 28, 28, 32, 192, 1, 1, 0, 0, 1, 1)),
    # ('workloads_70', Workload(1, 14, 14, 192, 192, 3, 3, 1, 1, 1, 1)),
    # ('workloads_71', Workload(1, 14, 14, 192, 32, 1, 1, 0, 0, 1, 1)),
    # ('workloads_72', Workload(1, 14, 14, 32, 192, 1, 1, 0, 0, 1, 1)),
    # ('workloads_73', Workload(1, 14, 14, 192, 192, 3, 3, 1, 1, 2, 2)),
    # ('workloads_74', Workload(1, 14, 14, 192, 64, 1, 1, 0, 0, 1, 1)),
    # ('workloads_75', Workload(1, 14, 14, 64, 384, 1, 1, 0, 0, 1, 1)),
    # ('workloads_76', Workload(1, 14, 14, 384, 384, 3, 3, 1, 1, 1, 1)),
    # ('workloads_77', Workload(1, 14, 14, 384, 64, 1, 1, 0, 0, 1, 1)),
    # ('workloads_78', Workload(1, 7, 7, 64, 384, 1, 1, 0, 0, 1, 1)),
    # ('workloads_79', Workload(1, 7, 7, 384, 384, 3, 3, 1, 1, 1, 1)),
    # ('workloads_80', Workload(1, 7, 7, 384, 64, 1, 1, 0, 0, 1, 1)),
    # ('workloads_81', Workload(1, 7, 7, 384, 96, 1, 1, 0, 0, 1, 1)),
    # ('workloads_82', Workload(1, 7, 7, 96, 576, 1, 1, 0, 0, 1, 1)),
    # ('workloads_83', Workload(1, 7, 7, 576, 576, 3, 3, 1, 1, 1, 1)),
    # ('workloads_84', Workload(1, 7, 7, 912, 912, 3, 3, 1, 1, 2, 2)),
    # ('workloads_85', Workload(1, 7, 7, 912, 912, 1, 1, 0, 0, 1, 1)),
    # ('workloads_86', Workload(1, 112, 112, 32, 32, 1, 1, 0, 0, 2, 2)),
    # ('workloads_87', Workload(1, 112, 112, 32, 32, 1, 1, 0, 0, 1, 1)),
    # ('workloads_88', Workload(1, 112, 112, 32, 32, 3, 3, 1, 1, 2, 2)),
    # ('workloads_89', Workload(1, 56, 56, 32, 32, 1, 1, 0, 0, 1, 1)),
    # ('workloads_90', Workload(1, 56, 56, 32, 64, 1, 1, 0, 0, 2, 2)),
    # ('workloads_91', Workload(1, 56, 56, 32, 64, 1, 1, 0, 0, 1, 1)),
    # ('workloads_92', Workload(1, 56, 56, 64, 64, 3, 3, 1, 1, 2, 2)),
    # ('workloads_93', Workload(1, 28, 28, 64, 64, 1, 1, 0, 0, 1, 1)),
    # ('workloads_94', Workload(1, 28, 28, 64, 64, 3, 3, 1, 1, 1, 1)),
    # ('workloads_95', Workload(1, 28, 28, 64, 160, 1, 1, 0, 0, 2, 2)),
    # ('workloads_96', Workload(1, 28, 28, 64, 160, 1, 1, 0, 0, 1, 1)),
    # ('workloads_97', Workload(1, 28, 28, 160, 160, 3, 3, 1, 1, 2, 2)),
    # ('workloads_98', Workload(1, 14, 14, 160, 160, 1, 1, 0, 0, 1, 1)),
    # ('workloads_99', Workload(1, 14, 14, 160, 160, 3, 3, 1, 1, 1, 1)),
    # ('workloads_100', Workload(1, 14, 14, 160, 400, 1, 1, 0, 0, 2, 2)),
    # ('workloads_101', Workload(1, 14, 14, 160, 400, 1, 1, 0, 0, 1, 1)),
    # ('workloads_102', Workload(1, 14, 14, 400, 400, 3, 3, 1, 1, 2, 2)),
    # ('workloads_103', Workload(1, 7, 7, 400, 400, 1, 1, 0, 0, 1, 1)),
    # ('workloads_104', Workload(1, 7, 7, 400, 400, 3, 3, 1, 1, 1, 1)),
    # ('workloads_105', Workload(1, 56, 56, 64, 128, 1, 1, 0, 0, 1, 1)),
    # ('workloads_106', Workload(1, 56, 56, 128, 128, 3, 3, 1, 1, 1, 1)),
    # ('workloads_107', Workload(1, 56, 56, 128, 256, 1, 1, 0, 0, 1, 1)),
    # ('workloads_108', Workload(1, 56, 56, 64, 256, 1, 1, 0, 0, 1, 1)),
    # ('workloads_109', Workload(1, 56, 56, 256, 128, 1, 1, 0, 0, 1, 1)),
    # ('workloads_110', Workload(1, 28, 28, 128, 128, 3, 3, 1, 1, 1, 1)),
    # ('workloads_111', Workload(1, 28, 28, 256, 256, 1, 1, 0, 0, 1, 1)),
    # ('workloads_112', Workload(1, 28, 28, 256, 256, 3, 3, 1, 1, 2, 2)),
    # ('workloads_113', Workload(1, 28, 28, 256, 512, 1, 1, 0, 0, 1, 1)),
    # ('workloads_114', Workload(1, 28, 28, 256, 512, 1, 1, 0, 0, 2, 2)),
    # ('workloads_115', Workload(1, 28, 28, 512, 256, 1, 1, 0, 0, 1, 1)),
    # ('workloads_116', Workload(1, 28, 28, 256, 256, 3, 3, 1, 1, 1, 1)),
    # ('workloads_117', Workload(1, 14, 14, 256, 256, 3, 3, 1, 1, 1, 1)),
    # ('workloads_118', Workload(1, 14, 14, 512, 256, 1, 1, 0, 0, 1, 1)),
    # ('workloads_119', Workload(1, 14, 14, 256, 512, 1, 1, 0, 0, 1, 1)),
    # ('workloads_120', Workload(1, 14, 14, 512, 512, 1, 1, 0, 0, 1, 1)),
    # ('workloads_121', Workload(1, 14, 14, 512, 512, 3, 3, 1, 1, 2, 2)),
    # ('workloads_122', Workload(1, 14, 14, 512, 1024, 1, 1, 0, 0, 1, 1)),
    # ('workloads_123', Workload(1, 14, 14, 512, 1024, 1, 1, 0, 0, 2, 2)),
    # ('workloads_124', Workload(1, 14, 14, 1024, 512, 1, 1, 0, 0, 1, 1)),
    # ('workloads_125', Workload(1, 14, 14, 512, 512, 3, 3, 1, 1, 1, 1)),
    # ('workloads_126', Workload(1, 7, 7, 512, 512, 3, 3, 1, 1, 1, 1)),
    # ('workloads_127', Workload(1, 7, 7, 1024, 512, 1, 1, 0, 0, 1, 1)),
    # ('workloads_128', Workload(1, 7, 7, 512, 1024, 1, 1, 0, 0, 1, 1)),
    # ('workloads_129', Workload(1, 7, 7, 1024, 1024, 1, 1, 0, 0, 1, 1)),
    # ('workloads_130', Workload(1, 7, 7, 1024, 1024, 3, 3, 1, 1, 2, 2)),
    # ('workloads_131', Workload(1, 7, 7, 1024, 2048, 1, 1, 0, 0, 1, 1)),
    # ('workloads_132', Workload(1, 55, 55, 64, 16, 1, 1, 0, 0, 1, 1)),
    # ('workloads_133', Workload(1, 55, 55, 16, 64, 1, 1, 0, 0, 1, 1)),
    # ('workloads_134', Workload(1, 55, 55, 16, 64, 3, 3, 1, 1, 1, 1)),
    # ('workloads_135', Workload(1, 55, 55, 128, 16, 1, 1, 0, 0, 1, 1)),
    # ('workloads_136', Workload(1, 55, 55, 128, 32, 1, 1, 0, 0, 1, 1)),
    # ('workloads_137', Workload(1, 27, 27, 32, 128, 1, 1, 0, 0, 1, 1)),
    # ('workloads_138', Workload(1, 27, 27, 32, 128, 3, 3, 1, 1, 1, 1)),
    # ('workloads_139', Workload(1, 27, 27, 256, 32, 1, 1, 0, 0, 1, 1)),
    # ('workloads_140', Workload(1, 27, 27, 256, 48, 1, 1, 0, 0, 1, 1)),
    # ('workloads_141', Workload(1, 13, 13, 48, 192, 1, 1, 0, 0, 1, 1)),
    # ('workloads_142', Workload(1, 13, 13, 48, 192, 3, 3, 1, 1, 1, 1)),
    # ('workloads_143', Workload(1, 13, 13, 384, 48, 1, 1, 0, 0, 1, 1)),
    # ('workloads_144', Workload(1, 13, 13, 384, 64, 1, 1, 0, 0, 1, 1)),
    # ('workloads_145', Workload(1, 13, 13, 64, 256, 1, 1, 0, 0, 1, 1)),
    # ('workloads_146', Workload(1, 13, 13, 64, 256, 3, 3, 1, 1, 1, 1)),
    # ('workloads_147', Workload(1, 13, 13, 512, 64, 1, 1, 0, 0, 1, 1)),
    # ('workloads_148', Workload(1, 112, 112, 64, 128, 3, 3, 1, 1, 1, 1)),
    # ('workloads_149', Workload(1, 56, 56, 128, 256, 3, 3, 1, 1, 1, 1)),
    # ('workloads_150', Workload(1, 56, 56, 256, 256, 3, 3, 1, 1, 1, 1)),
    # ('workloads_151', Workload(1, 28, 28, 256, 512, 3, 3, 1, 1, 1, 1)),
    # ('workloads_152', Workload(1, 28, 28, 512, 512, 3, 3, 1, 1, 1, 1)),
    # ('workloads_153', Workload(1, 224, 224, 64, 64, 3, 3, 1, 1, 1, 1)),
    # ('workloads_154', Workload(1, 112, 112, 128, 128, 3, 3, 1, 1, 1, 1)),
    # ('workloads_155', Workload(1, 56, 56, 256, 256, 1, 1, 0, 0, 1, 1)),
    # ('workloads_156', Workload(1, 56, 56, 256, 256, 3, 3, 1, 1, 2, 2)),
    # ('workloads_157', Workload(1, 56, 56, 256, 512, 1, 1, 0, 0, 2, 2)),
    # ('workloads_158', Workload(1, 28, 28, 512, 512, 1, 1, 0, 0, 1, 1)),
    # ('workloads_159', Workload(1, 28, 28, 512, 512, 3, 3, 1, 1, 2, 2)),
    # ('workloads_160', Workload(1, 28, 28, 512, 1024, 1, 1, 0, 0, 2, 2)),
    # ('workloads_161', Workload(1, 14, 14, 1024, 1024, 1, 1, 0, 0, 1, 1)),
    # ('workloads_162', Workload(1, 14, 14, 1024, 1024, 3, 3, 1, 1, 2, 2)),
    # ('workloads_163', Workload(1, 14, 14, 1024, 2048, 1, 1, 0, 0, 2, 2)),
    # ('workloads_164', Workload(1, 7, 7, 2048, 1024, 1, 1, 0, 0, 1, 1)),
    # ('workloads_165', Workload(1, 7, 7, 1024, 1024, 3, 3, 1, 1, 1, 1)),
    ('workloads_0', Workload(1, 149, 149, 32, 32, 3, 3, 0, 0, 1, 1)),
    ('workloads_1', Workload(1, 147, 147, 32, 64, 3, 3, 1, 1, 1, 1)),
    ('workloads_2', Workload(1, 73, 73, 64, 80, 1, 1, 0, 0, 1, 1)),
    ('workloads_3', Workload(1, 73, 73, 80, 192, 3, 3, 0, 0, 1, 1)),
    ('workloads_4', Workload(1, 35, 35, 192, 64, 1, 1, 0, 0, 1, 1)),
    ('workloads_5', Workload(1, 35, 35, 192, 48, 1, 1, 0, 0, 1, 1)),
    ('workloads_6', Workload(1, 35, 35, 48, 64, 5, 5, 2, 2, 1, 1)),
    ('workloads_7', Workload(1, 35, 35, 64, 96, 3, 3, 1, 1, 1, 1)),
    ('workloads_8', Workload(1, 35, 35, 96, 96, 3, 3, 1, 1, 1, 1)),
    ('workloads_9', Workload(1, 35, 35, 192, 32, 1, 1, 0, 0, 1, 1)),
    ('workloads_10', Workload(1, 35, 35, 256, 64, 1, 1, 0, 0, 1, 1)),
    ('workloads_11', Workload(1, 35, 35, 256, 48, 1, 1, 0, 0, 1, 1)),
    ('workloads_12', Workload(1, 35, 35, 288, 64, 1, 1, 0, 0, 1, 1)),
    ('workloads_13', Workload(1, 35, 35, 288, 48, 1, 1, 0, 0, 1, 1)),
    ('workloads_14', Workload(1, 35, 35, 288, 384, 3, 3, 0, 0, 2, 2)),
    ('workloads_15', Workload(1, 35, 35, 96, 96, 3, 3, 0, 0, 2, 2)),
    ('workloads_16', Workload(1, 17, 17, 768, 192, 1, 1, 0, 0, 1, 1)),
    ('workloads_17', Workload(1, 17, 17, 768, 128, 1, 1, 0, 0, 1, 1)),
    ('workloads_18', Workload(1, 17, 17, 768, 160, 1, 1, 0, 0, 1, 1)),
    ('workloads_19', Workload(1, 17, 17, 128, 768, 5, 5, 0, 0, 1, 1)),
    ('workloads_20', Workload(1, 17, 17, 192, 320, 3, 3, 0, 0, 2, 2)),
    ('workloads_21', Workload(1, 8, 8, 192, 192, 3, 3, 0, 0, 2, 2)),
    ('workloads_22', Workload(1, 8, 8, 1280, 320, 1, 1, 0, 0, 1, 1)),
    ('workloads_23', Workload(1, 8, 8, 1280, 384, 1, 1, 0, 0, 1, 1)),
    ('workloads_24', Workload(1, 8, 8, 1280, 448, 1, 1, 0, 0, 1, 1)),
    ('workloads_25', Workload(1, 8, 8, 448, 384, 3, 3, 1, 1, 1, 1)),
    ('workloads_26', Workload(1, 8, 8, 1280, 192, 1, 1, 0, 0, 1, 1)),
    ('workloads_27', Workload(1, 8, 8, 2048, 320, 1, 1, 0, 0, 1, 1)),
    ('workloads_28', Workload(1, 8, 8, 2048, 384, 1, 1, 0, 0, 1, 1)),
    ('workloads_29', Workload(1, 8, 8, 2048, 448, 1, 1, 0, 0, 1, 1)),

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


def run_relu(env, remote, wl, target, check_correctness=True, print_ir=False, samples=4, log_file='logs/log.json', host_ip = '192.168.2.99'):

    # Workload assertions
    assert wl.hpad == wl.wpad

    if not os.path.exists(log_file):
        with open(log_file, 'w+') as myfile:
            json.dump({"workloads": []}, myfile, indent=4)

    CPU_exec = nn.Sequential(nn.Conv2d(3, wl.in_filter, kernel_size=(3,3), stride=(1,1),
                                         padding=(1,1)), nn.BatchNorm2d(wl.in_filter), nn.ReLU(inplace=True))

    vta_exec_conv = nn.Conv2d(wl.in_filter, wl.out_filter, kernel_size=(wl.hkernel, wl.wkernel),
                              stride=(wl.hstride, wl.wstride), padding=(wl.hpad, wl.wpad))

    vta_exec_batchnorm = nn.BatchNorm2d(wl.out_filter)
    vta_exec_relu = nn.ReLU()

    last_layer = nn.AdaptiveAvgPool2d((1,1))

    model = nn.Sequential(CPU_exec, vta_exec_conv, last_layer)

    # Load pre-configured AutoTVM schedules
    with autotvm.tophub.context(target, extra_files=schedule_log_files):

        input_name = "input0"

        # Populate the shape and data type dictionary for ImageNet classifier input
        dtype_dict = {input_name: "float32"}
        shape_dict = {input_name: (env.BATCH, 3, wl.height, wl.width)}

        input_shape = [env.BATCH, 3, wl.height, wl.width]
        input_data = torch.randn(input_shape)

        scripted_model = torch.jit.trace(model, input_data).eval()
        shape_list = [(input_name, input_shape)]


        mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
        shape_dict.update({k: v.shape for k, v in params.items()})
        dtype_dict.update({k: str(v.dtype) for k, v in params.items()})

        if target.device_name == "vta":
            # Perform quantization in Relay
            # Note: We set opt_level to 3 in order to fold batch norm
            with tvm.transform.PassContext(opt_level=3):
                with relay.quantize.qconfig(global_scale=8.0, skip_conv_layers=[0]):
                    mod = relay.quantize.quantize(mod, params=params)

                    # print(mod.astext(show_meta_data=False))
                    # exit(0)

                # Perform graph packing and constant folding for VTA target
                assert env.BLOCK_IN == env.BLOCK_OUT
                # do device annotation if target is intelfocl or sim
                relay_prog = graph_pack(
                    mod["main"],
                    env.BATCH,
                    env.BLOCK_OUT,
                    env.WGT_WIDTH,
                    start_name='cast',
                    start_name_idx=8,
                    stop_name='nn.adaptive_avg_pool2d',
                    # start_name_idx=6,
                    # start_name='nn.max_pool2d',
                    device_annot=(env.TARGET == "intelfocl"),
                )
        else:
            relay_prog = mod["main"]

        with vta.build_config(opt_level=3, disabled_pass={"AlterOpLayout"}):
            graph, lib, params = relay.build(
                relay_prog, target=target, params=params, target_host=env.target_host
            )


        temp = utils.tempdir()
        lib.export_library(temp.relpath("graphlib.tar"))
        remote.upload(temp.relpath("graphlib.tar"))
        lib = remote.load_module("graphlib.tar")

        ctx = remote.ext_dev(0)
        m = graph_executor.create(graph, lib, ctx)

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
    m.set_input(**params)
    m.set_input(input_name, input_data)
    for slot in range(6):
        per_slot_dict = {"samples": [], "overall": {"write_bytes": 0, "read_bytes": 0}}
        #myfile.write("\nSlot {} ".format(slot))

        print("starting polling subprocess")
        proc = subprocess.Popen(["sshpass", "-p", "Srivat95", "ssh", "-t", "xilinx@{}".format(host_ip), "sudo", "python3",
                             "/home/xilinx/tvm/vta/python/vta/poll_apm.py", "--slot", str(slot)], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
        count = 0
        for i in range(200000000):
            count += 1
        m.run()

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

        print("Ran batch norm successfully!!")
    workload_dict["results"] = result_dict
    with open(log_file, 'r+') as myfile:
        file_data = json.load(myfile)
        file_data["workloads"].append(workload_dict)
        myfile.seek(0)
        json.dump(file_data, myfile, indent=4)

def test_relu(device, log_file = "logs/log.json", host_ip = '192.168.2.99'):
    #device_host = os.environ.get("VTA_RPC_HOST", "192.168.2.99")

    device_host = host_ip

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
            run_relu(env, remote, wl, target, log_file=log_file, host_ip=host_ip)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AXI Performance Monitor Convolution Benchmark')
    parser.add_argument('--log_file', type=str, default="logs/log.json",
                        help='output log file path')
    parser.add_argument('--host_ip', type=str, default='192.168.2.99',
                        help='pynq board IP')

    args = parser.parse_args()
    #test_conv2d(device="arm_cpu")
    test_relu(device="vta", log_file = args.log_file, host_ip = args.host_ip)


