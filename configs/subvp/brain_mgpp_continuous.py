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
"""Training NCSN++ on ABCD brains with sub-VP SDE."""
from configs.default_brain_configs import get_default_configs


def get_config():
    config = get_default_configs()
    # training
    training = config.training
    training.sde = "subvpsde"
    training.continuous = True
    training.reduce_mean = True
    training.batch_size = 2

    config.data.image_size = (112, 136, 104)
    config.data.spacing_pix_dim = 1.5

    # sampling
    sampling = config.sampling
    sampling.method = "pc"
    sampling.predictor = "euler_maruyama"
    sampling.corrector = "none"

    # model
    model = config.model
    model.name = "models_genesis_pp"
    model.ema_rate = 0.9999
    model.time_embedding_sz = 1024 # After dense layers
    model.init_scale = 0.0
    model.fourier_scale = 16
    model.nonlinearity = "elu"
    
    config.optim.lr = 1e-5
    config.optim.weight_decay = 0.0
    
    return config
