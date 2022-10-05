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
from configs.ve.brain_ncsnpp_shallow import get_config as get_orig_config


def get_config():
    config = get_orig_config()
    # training
    training = config.training
    training.batch_size = 2
    training.n_iters = 1000001
    training.load_pretrain = True
    training.pretrain_dir = "/ahsan_projects/braintypicality/workdir/brain_ve_test/shallow/checkpoints-meta"
    training.eval_freq = 1000

    data = config.data
    data.image_size = (192, 224, 160)  # For generating images
    data.spacing_pix_dim = 1.0

    evaluate = config.eval
    evaluate.sample_size = 4
    evaluate.batch_size = 4

    # optimization
    optim = config.optim
    optim.weight_decay = 0.0
    optim.optimizer = "Adam"
    optim.lr = 2e-4
    optim.warmup = 5000
    optim.scheduler = "skip"

    model = config.model
    model.sigma_max = 772.0

    return config
