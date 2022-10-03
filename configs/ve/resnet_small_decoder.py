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
"""Training NCSN++ on Brains with VE SDE.
   Keeping it consistent with CelebaHQ config from Song
"""
from configs.default_brain_configs import get_default_configs


def get_config():
    config = get_default_configs()
    # training
    training = config.training
    training.sde = "vesde"
    training.continuous = True
    training.likelihood_weighting = False
    training.reduce_mean = False
    training.batch_size = 8
    training.n_iters = 1500001

    data = config.data
    data.image_size = (96, 128, 96)
    data.spacing_pix_dim = 2.0
    data.num_channels = 1
    data.select_channel = 1
    data.cache_rate = 1.0
    data.centered = False

    evaluate = config.eval
    evaluate.sample_size = 8

    # optimization
    optim = config.optim
    optim.weight_decay = 0.0
    optim.optimizer = "Adam"
    optim.lr = 2e-4
    optim.warmup = 5000
    optim.scheduler = "skip"

    # sampling
    sampling = config.sampling
    sampling.method = "pc"
    sampling.predictor = "reverse_diffusion"
    sampling.corrector = "langevin"
    sampling.probability_flow = False
    sampling.snr = 0.17
    sampling.n_steps_each = 1
    sampling.noise_removal = True

    # model
    model = config.model
    model.name = "ncsnpp3d"
    model.act = "memswish"
    model.scale_by_sigma = True
    model.ema_rate = 0.9999
    model.nf = 16
    model.blocks_down = (2, 4, 4, 8)
    model.blocks_up = (1, 1, 1)
    model.time_embedding_sz = 128
    model.init_scale = 0.0
    model.fourier_scale = 16.0
    model.num_scales = 2000
    model.conv_size = 3
    model.attention_heads = 0
    model.dropout = 0.0
    model.resblock_pp = True
    model.embedding_type = "fourier"
    model.dilation = 1

    return config
