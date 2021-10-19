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
"""Training NCSN++ on Brains with VE SDE."""
from configs.default_brain_configs import get_default_configs


def get_config():
    config = get_default_configs()
    # training
    training = config.training
    training.sde = "vesde"
    training.continuous = True
    training.likelihood_weighting = False
    training.reduce_mean = True

    # sampling
    sampling = config.sampling
    sampling.method = "pc"
    sampling.predictor = "reverse_diffusion"
    sampling.corrector = "langevin"
    sampling.probability_flow = False
    sampling.snr = 0.15
    sampling.n_steps_each = 1
    sampling.noise_removal = True

    # model
    model = config.model
    model.name = "ncsnpp3d"
    model.scale_by_sigma = False
    model.ema_rate = 0.9999
    model.nf = 16
    model.ch_mult = (1, 2, 2, 2)
    model.num_res_blocks = 4
    model.attn_resolutions = (16,)
    model.time_embedding_sz = 512
    model.init_scale = 0.0
    model.fourier_scale = 16
    model.conv_size = 3

    return config
