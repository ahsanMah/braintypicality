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

"""Training and evaluation"""
import os

# os.environ["WANDB_START_METHOD"] = "thread"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# conda config --set auto_activate_base false

import run_lib
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import logging
import os
import tensorflow as tf
import wandb
import ml_collections
from collections import defaultdict
from pprint import pprint

# os.environ["WANDB_RUN_ID"] = "ve-tests"  # wandb.util.generate_id()
# os.symlink("/DATA/", "/BEE/Connectome/ABCD/")
# Add a symlink

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow from using GPU
  try:
    tf.config.experimental.set_visible_devices([], 'GPU')
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)

import warnings

warnings.filterwarnings("ignore")

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=True
)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_enum(
    "mode", None, ["train", "eval", "score", "sweep"], "Running mode: train or eval"
)
flags.DEFINE_string(
    "eval_folder", "eval", "The folder name for storing evaluation results"
)
flags.DEFINE_string(
    "sweep_id", None, "Optional ID for a sweep controller if running a sweep."
)
flags.DEFINE_string("project", None, "Wandb project name.")
# flags.DEFINE_string("pretrain_dir", None, "Directory with pretrained weights.")
flags.mark_flags_as_required(["workdir", "config", "mode", "project"])


def main(argv):

    if FLAGS.mode == "sweep":

        def train_sweep():

            with wandb.init():
                # Process config params to ml dict
                params = dict()
                config = FLAGS.config.to_dict()
                sweep_config = wandb.config

                for p, val in sweep_config.items():
                    # First '_' splits into upper level
                    keys = p.split("_")
                    # print(keys)
                    parent = keys[0]
                    child = "_".join(keys[1:])
                    config[parent][child] = val

                wandb.config.update(config)
                config = ml_collections.ConfigDict(wandb.config)

                # Create the working directory
                FLAGS.workdir = FLAGS.workdir + f"/{wandb.run.name}"
                tf.io.gfile.makedirs(FLAGS.workdir)

                # Set logger so that it outputs to both console and file
                # Make logging work for both disk and Google Cloud Storage
                gfile_stream = open(os.path.join(FLAGS.workdir, "stdout.txt"), "w")
                handler = logging.StreamHandler(gfile_stream)
                formatter = logging.Formatter(
                    "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
                )
                handler.setFormatter(formatter)
                logger = logging.getLogger()
                logger.addHandler(handler)
                logger.setLevel("INFO")
                # Run the training pipeline
                run_lib.train(config, FLAGS.workdir)
            return

        config = FLAGS.config.to_dict()
        sweep_config = config["sweep"]
        print(sweep_config)
        # FIXME: this way intitalizes a new sweep copntroller each time
        # Put this in a separate script so that we only init the master controller once
        if FLAGS.sweep_id is not None:
            sweep_id = FLAGS.sweep_id
        else:
            sweep_id = wandb.sweep(sweep_config, project="braintyp")

        print("Sweep ID:", sweep_id, type(sweep_id))

        wandb.agent(sweep_id, train_sweep, project="braintyp", count=10)

    elif FLAGS.mode == "train":

        with wandb.init(
            project=FLAGS.project, config=FLAGS.config.to_dict(), resume="allow"
        ):

            config = ml_collections.ConfigDict(wandb.config)

            # Create the working directory
            tf.io.gfile.makedirs(FLAGS.workdir)
            # Set logger so that it outputs to both console and file
            # Make logging work for both disk and Google Cloud Storage
            gfile_stream = open(os.path.join(FLAGS.workdir, "stdout.txt"), "w")
            handler = logging.StreamHandler(gfile_stream)
            formatter = logging.Formatter(
                "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger = logging.getLogger()
            logger.addHandler(handler)
            logger.setLevel("INFO")
            # Run the training pipeline
            run_lib.train(config, FLAGS.workdir)

    elif FLAGS.mode == "eval":
        # Run the evaluation pipeline
        run_lib.evaluate(FLAGS.config, FLAGS.workdir, FLAGS.eval_folder)
    elif FLAGS.mode == "score":
        # Run the evaluation pipeline
        run_lib.compute_scores(FLAGS.config, FLAGS.workdir)
    else:
        raise ValueError(f"Mode {FLAGS.mode} not recognized.")


if __name__ == "__main__":
    app.run(main)
