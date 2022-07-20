
"""
Auto-tuning a convolutional network on VTA
==========================================
**Author**: `Lianmin Zheng <https://github.com/merrymercy>`_, `Thierry Moreau <https://homes.cs.washington.edu/~moreau/>`_

Auto-tuning for a specific accelerator design is critical for getting the best
performance for any given operator. This is a tutorial showcases how to tune a
whole convolutional network on VTA.

The operator implementation for VTA in TVM is written in template form.
The template has many tunable knobs (tile factor, virtual threads, etc).
We will tune all convolution operators in the neural network. After tuning,
we produce a log file which stores the best schedule parameters for all tuned
operators. When the TVM compiler compiles these operators, it will query this
log file to get the best knob parameters.

"""


import os

from mxnet.gluon.model_zoo import vision
import numpy as np
from PIL import Image

from tvm import topi
import tvm
from tvm import te
from tvm import rpc, autotvm, relay
from tvm.contrib import graph_executor, utils, download
from tvm.autotvm.measure.measure_methods import request_remote
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner

import vta
from vta.testing import simulator
from vta.top import graph_pack

import torch

#################################################################
# Compile network
# ---------------
# Perform vta-specific compilation with Relay from a Gluon model


def compile_network_pytorch(env, target, model, start_pack, start_pack_idx, stop_pack, stop_pack_idx):
    input_name = "input0"

    # Populate the shape and data type dictionary
    dtype_dict = {"data": "float32"}
    shape_dict = {"data": (env.BATCH, 3, 224, 224)}

    # Get off the shelf gluon model, and convert to relay
    pytorch_model = torch.hub.load('pytorch/vision', model, pretrained=True)
    input_shape = [1, 3, 224, 224]
    input_data = torch.randn(input_shape)
    scripted_model = torch.jit.trace(pytorch_model, input_data).eval()

    shape_list = [(input_name, input_shape)]
    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

    # Update shape and type dictionary
    shape_dict.update({k: v.shape for k, v in params.items()})
    dtype_dict.update({k: str(v.dtype) for k, v in params.items()})

    # Perform quantization in Relay
    # Note: We set opt_level to 3 in order to fold batch norm
    with tvm.transform.PassContext(opt_level=3):
        with relay.quantize.qconfig(global_scale=8.0, skip_conv_layers=[0]):
            mod = relay.quantize.quantize(mod, params=params)
    # print(mod.astext(show_meta_data=False))
    # exit(0)

    # Perform graph packing and constant folding for VTA target
    if target.device_name == "vta":
        assert env.BLOCK_IN == env.BLOCK_OUT
        relay_prog = graph_pack(
            mod["main"],
            env.BATCH,
            env.BLOCK_OUT,
            env.WGT_WIDTH,
            start_name=start_pack,
            start_name_idx=start_pack_idx,
            stop_name=stop_pack,
            stop_name_idx=stop_pack_idx,
        )

    return relay_prog, params


def compile_network_mxnet(env, target, model, start_pack, start_pack_idx, stop_pack, stop_pack_idx):
    # Populate the shape and data type dictionary
    dtype_dict = {"data": "float32"}
    shape_dict = {"data": (env.BATCH, 3, 224, 224)}

    # Get off the shelf gluon model, and convert to relay
    gluon_model = vision.get_model(model, pretrained=True)
    mod, params = relay.frontend.from_mxnet(gluon_model, shape_dict)

    # Update shape and type dictionary
    shape_dict.update({k: v.shape for k, v in params.items()})
    dtype_dict.update({k: str(v.dtype) for k, v in params.items()})

    # Perform quantization in Relay
    # Note: We set opt_level to 3 in order to fold batch norm
    with tvm.transform.PassContext(opt_level=3):
        with relay.quantize.qconfig(global_scale=8.0, skip_conv_layers=[0, 25, 31]):
            mod = relay.quantize.quantize(mod, params=params)



    # Perform graph packing and constant folding for VTA target
    if target.device_name == "vta":
        assert env.BLOCK_IN == env.BLOCK_OUT
        relay_prog = graph_pack(
            mod["main"],
            env.BATCH,
            env.BLOCK_OUT,
            env.WGT_WIDTH,
            start_name=start_pack,
            start_name_idx=start_pack_idx,
            stop_name=stop_pack,
            stop_name_idx=stop_pack_idx,
        )

    return relay_prog, params


# Tracker host and port can be set by your environment
tracker_host = os.environ.get("TVM_TRACKER_HOST", "127.0.0.1")
tracker_port = int(os.environ.get("TVM_TRACKER_PORT", 9190))

# Load VTA parameters from the 3rdparty/vta-hw/config/vta_config.json file
env = vta.get_env()

# This target is used for cross compilation. You can query it by :code:`gcc -v` on your device.
# Set ``device=arm_cpu`` to run inference on the CPU
# or ``device=vta`` to run inference on the FPGA.
device = "vta"
target = env.target if device == "vta" else env.target_vta_cpu

# Name of Gluon model to compile
# The ``start_pack`` and ``stop_pack`` labels indicate where
# to start and end the graph packing relay pass: in other words
# where to start and finish offloading to VTA.
network = "vgg19"
start_pack = "cast"
start_pack_idx=7
stop_pack = "cast"
stop_pack_idx = 136

# Tuning option
log_file = "logs/%s.%s.log" % (device, network)
tuning_option = {
    "log_filename": log_file,
    "tuner": "random",
    "n_trial": 1000,
    "early_stopping": None,
    "measure_option": autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.RPCRunner(
            env.TARGET,
            host=tracker_host,
            port=tracker_port,
            number=5,
            timeout=60,
            module_loader=vta.module_loader(),
            # check_correctness=True, # TODO: re-enable when check_correctness works again.
        ),
    ),
}

####################################################################
#
# .. note:: How to set tuning options
#
#   In general, the default values provided here work well.
#   If you have enough time budget, you can set :code:`n_trial`, :code:`early_stopping`
#   to larger values, makes the tuning run for longer.
#   If your device is under-powered or your conv2d operators are large, consider
#   setting a longer timeout.
#

###################################################################
# Begin Tuning
# ------------
# Now we can extract tuning tasks from the network and begin tuning.
# Here, we provide a simple utility function to tune a list of tasks.
# This function is just an initial implementation which tunes them in sequential order.
# We will introduce a more sophisticated tuning scheduler in the future.
#
# Given that the tuning will be done on Pynq FPGA boards, make sure that
# the ```TARGET`` entry in the ``vta_config.json`` file is set to ``pynq``.


# You can skip the implementation of this function for this tutorial.
def tune_tasks(
    tasks,
    measure_option,
    tuner="xgb",
    n_trial=1000,
    early_stopping=None,
    log_filename="tuning.log",
    use_transfer_learning=True,
):

    # create tmp log file
    tmp_log_file = log_filename + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

        # create tuner
        if tuner == "xgb" or tuner == "xgb-rank":
            tuner_obj = XGBTuner(tsk, loss_type="rank")
        elif tuner == "xgb_knob":
            tuner_obj = XGBTuner(tsk, loss_type="rank", feature_type="knob")
        elif tuner == "ga":
            tuner_obj = GATuner(tsk, pop_size=50)
        elif tuner == "random":
            tuner_obj = RandomTuner(tsk)
        elif tuner == "gridsearch":
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        if use_transfer_learning:
            if os.path.isfile(tmp_log_file):
                tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))

        # do tuning
        tsk_trial = min(n_trial, len(tsk.config_space))
        tuner_obj.tune(
            n_trial=tsk_trial,
            early_stopping=early_stopping,
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
                autotvm.callback.log_to_file(tmp_log_file),
            ],
        )

    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)
    os.remove(tmp_log_file)


########################################################################
# Register VTA-specific tuning tasks


def register_vta_tuning_tasks():
    from tvm.autotvm.task import TaskExtractEnv

    @tvm.te.tag_scope(tag=topi.tag.ELEMWISE)
    def my_clip(x, a_min, a_max):
        """Unlike topi's current clip, put min and max into two stages."""
        const_min = tvm.tir.const(a_min, x.dtype)
        const_max = tvm.tir.const(a_max, x.dtype)
        x = te.compute(x.shape, lambda *i: tvm.te.min(x(*i), const_max), name="clipA")
        x = te.compute(x.shape, lambda *i: tvm.te.max(x(*i), const_min), name="clipB")
        return x

    # init autotvm env to register VTA operator
    TaskExtractEnv()

    @autotvm.template("conv2d_packed.vta")
    def _topi_nn_conv2d(*args, **kwargs):
        assert not kwargs, "Do not support kwargs in template function call"
        A, W = args[:2]

        with tvm.target.vta():
            res = vta.top.conv2d_packed(*args, **kwargs)
            res = topi.right_shift(res, 8)
            res = my_clip(res, 0, 127)
            res = topi.cast(res, "int8")

        if tvm.target.Target.current().device_name == "vta":
            s = vta.top.schedule_conv2d_packed([res])
        else:
            s = te.create_schedule([res.op])
        return s, [A, W, res]


########################################################################
# Finally, we launch tuning jobs and evaluate the end-to-end performance.


def tune_and_evaluate(tuning_opt):

    # Register VTA tuning tasks
    register_vta_tuning_tasks()

    # Perform task extraction on Relay program
    print("Extract tasks...")
    relay_prog, params = compile_network_pytorch(env, target, network, start_pack, start_pack_idx, stop_pack, stop_pack_idx)
    mod = tvm.IRModule.from_expr(relay_prog)
    tasks = autotvm.task.extract_from_program(
        mod,
        params=params,
        ops=(relay.op.get("nn.conv2d"),),
        target=target,
        target_host=env.target_host,
    )

    # filter out non-packed conv2d task
    tasks = list(filter(lambda t: len(t.args[0][1]) > 4 and "conv" in t.name, tasks))

    print(tasks)
    exit(0)

    # We should have extracted 10 convolution tasks
    assert len(tasks) == 8
    print("Extracted {} conv2d tasks:".format(len(tasks)))
    for tsk in tasks:
        inp = tsk.args[0][1]
        wgt = tsk.args[1][1]
        batch = inp[0] * inp[4]
        in_filter = inp[1] * inp[5]
        out_filter = wgt[0] * wgt[4]
        height, width = inp[2], inp[3]
        hkernel, wkernel = wgt[2], wgt[3]
        hstride, wstride = tsk.args[2][0], tsk.args[2][1]
        hpad, wpad = tsk.args[3][0], tsk.args[3][1]
        print(
            "({}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {})".format(
                batch,
                height,
                width,
                in_filter,
                out_filter,
                hkernel,
                wkernel,
                hpad,
                wpad,
                hstride,
                wstride,
            )
        )

    # We do not run the tuning in our webpage server since it takes too long.
    # Comment the following line to run it by yourself.
    #return

    # run tuning tasks
    print("Tuning...")
    tune_tasks(tasks, **tuning_opt)

    # evaluate with tuning history
    if env.TARGET != "sim":
        # Get remote from fleet node
        remote = autotvm.measure.request_remote(
            env.TARGET, tracker_host, tracker_port, timeout=10000
        )
        # Reconfigure the JIT runtime and FPGA.
        vta.reconfig_runtime(remote)
        vta.program_fpga(remote, bitstream=None)
    else:
        # In simulation mode, host the RPC server locally.
        remote = rpc.LocalSession()

    # compile kernels with history best records
    with autotvm.tophub.context(target, extra_files=[log_file]):
        # Compile network
        print("Compile...")
        if target.device_name != "vta":
            with tvm.transform.PassContext(opt_level=3, disabled_pass={"AlterOpLayout"}):
                lib = relay.build(
                    relay_prog, target=target, params=params, target_host=env.target_host
                )
        else:
            with vta.build_config(opt_level=3, disabled_pass={"AlterOpLayout"}):
                lib = relay.build(
                    relay_prog, target=target, params=params, target_host=env.target_host
                )

        # Export library
        print("Upload...")
        temp = utils.tempdir()
        lib.export_library(temp.relpath("graphlib.tar"))
        remote.upload(temp.relpath("graphlib.tar"))
        lib = remote.load_module("graphlib.tar")

        # Generate the graph executor
        ctx = remote.ext_dev(0) if device == "vta" else remote.cpu(0)
        m = graph_executor.GraphModule(lib["default"](ctx))

        # upload parameters to device
        image = tvm.nd.array((np.random.uniform(size=(1, 3, 224, 224))).astype("float32"))
        m.set_input("input0", image)

        # evaluate
        print("Evaluate inference time cost...")
        timer = m.module.time_evaluator("run", ctx, number=1, repeat=10)
        tcost = timer()
        prof_res = np.array(tcost.results) * 1000  # convert to millisecond
        print(
            "Mean inference time (std dev): %.2f ms (%.2f ms)"
            % (np.mean(prof_res), np.std(prof_res))
        )


# Run the tuning and evaluate the results
tune_and_evaluate(tuning_option)
