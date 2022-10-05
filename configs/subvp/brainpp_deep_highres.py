# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Training NCSN++ on CIFAR-10 with sub-VP SDE."""
from configs.default_brain_configs import get_default_configs


def get_config():
    config = get_default_configs()
    # training
    training = config.training
    training.sde = "subvpsde"
    training.continuous = True
    training.reduce_mean = True
    training.batch_size = 1
    training.log_freq = 50
    training.eval_freq = 100
    training.n_iters = 950001
    training.sampling_freq = 30000

    data = config.data
    data.num_channels = 1
    data.select_channel = 1
    data.cache_rate = 1.0
    data.centered = False
    
    config.eval.batch_size = 32
    config.eval.sample_size = 8
    config.eval.enable_loss=False
    config.eval.enable_sampling = True
    config.eval.begin_ckpt = 10
    config.eval.end_ckpt = 10

    # sampling
    sampling = config.sampling
    sampling.method = "pc"
    sampling.predictor = "euler_maruyama"
    sampling.corrector = "none"

    # optim
    optim = config.optim
    optim.lr = 1e-4

    # model
    model = config.model
    model.name = "ncsnpp3d"
    model.scale_by_sigma = False
    model.ema_rate = 0.9999
    model.nf = 32
    model.blocks_down = (4, 4, 4, 8)
    model.blocks_up = (2, 2, 2)
    model.time_embedding_sz = 512
    model.init_scale = 0.0
    model.fourier_scale = 16.0
    model.num_scales = 4000
    model.conv_size = 3
    model.attention_heads = None
    model.dropout = 0.0
    model.resblock_pp = True
    model.embedding_type = "fourier"
    model.dilation = 1

    msma = config.msma
    msma.n_timesteps = 100

    # Configuration for Hyperparam sweeps
    sweep = config.sweep
    param_dict = dict(
        training_n_iters={"value": 50001},
        training_batch_size={"value": 4},
        eval_batch_size={"value": 32},
        training_snapshot_freq={"value": 10000},
        training_snapshot_freq_for_preemption={"value": 200000},
        model_dilation={"values": [1, 2]},
        model_dropout={"values": [0.0, 0.2]},
        model_embedding_type={"values": ["fourier", "positional"]},
        model_attention_heads={"values": [1, None]},
    )

    sweep.parameters = param_dict
    sweep.method = "random"
    sweep.metric = dict(name="val_loss")

    return config
