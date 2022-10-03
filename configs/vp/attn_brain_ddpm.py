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
    config.colab = False
    # training
    training = config.training
    training.sde = "vpsde"
    training.continuous = False
    training.reduce_mean = True
    training.batch_size = 32
    training.log_freq = 50
    training.eval_freq = 100

    config.eval.batch_size = 16

    data = config.data
    data.spacing_pix_dim = 4.0
    data.image_size = (48, 56, 40)
    data.cache_rate = 1.0
    data.centered = True

    # sampling
    sampling = config.sampling
    sampling.method = "pc"
    sampling.predictor = "ancestral_sampling"
    sampling.corrector = "none"

    # model
    model = config.model
    model.name = "ddpm3d"
    model.activation = "swish"
    model.scale_by_sigma = False
    model.ema_rate = 0.9999
    model.nf = 32
    # Add blocks down and blocks up tuples?
    model.time_embedding_sz = 1024
    model.init_scale = 0.0
    model.fourier_scale = 16
    model.conv_size = 3
    model.attention_heads = 4
    model.beta_min = 1e-4 / 4
    model.beta_max = 2e-2 / 4
    model.num_scales = 4000
    model.dropout = 0.1

    # config.optim.grad_clip = -1
    # param_dict = dict(
    #     model_

    #     training_n_iters={"value": 1001},
    #     training_log_freq={"value": 50},
    #     training_eval_freq={"value": 100},
    #     training_snapshot_freq={"value": 1000000},
    #     training_snapshot_freq_for_preemption={"value": 10000},
    # )

    return config
