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
    training.batch_size = 8
    training.log_freq = 50
    training.eval_freq = 100
    training.n_iters = 500001

    data = config.data
    data.image_size = (88, 104, 80)
    data.spacing_pix_dim = 2.0
    data.num_channels = 1
    data.select_channel = 1
    data.cache_rate = 1.0

    config.eval.batch_size = 16
    config.eval.sample_size = 8

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
    model.blocks_down = (2, 4, 4, 8)
    model.time_embedding_sz = 512
    model.init_scale = 0.0
    model.fourier_scale = 16.0
    model.conv_size = 3
    model.attention_heads = None
    model.dropout = 0.0

    msma = config.msma
    msma.n_timesteps = 100

    return config
