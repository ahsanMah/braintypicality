import ml_collections
import torch
import math


def get_default_configs():
    config = ml_collections.ConfigDict()
    # training
    config.training = training = ml_collections.ConfigDict()
    config.training.batch_size = 3
    training.n_iters = 250001
    training.snapshot_freq = 10001
    training.log_freq = 100
    training.eval_freq = 500
    ## store additional checkpoints for preemption in cloud computing environments
    training.snapshot_freq_for_preemption = 1000
    ## produce samples at each snapshot.
    training.snapshot_sampling = True
    training.likelihood_weighting = False
    training.continuous = True
    training.reduce_mean = False

    # sampling
    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.n_steps_each = 1
    sampling.noise_removal = True
    sampling.probability_flow = False
    sampling.snr = 0.075

    # evaluation
    config.eval = evaluate = ml_collections.ConfigDict()
    evaluate.begin_ckpt = 3
    evaluate.end_ckpt = 3
    evaluate.batch_size = 8
    evaluate.enable_sampling = False
    evaluate.num_samples = 50000
    evaluate.enable_loss = True
    evaluate.enable_bpd = True
    evaluate.bpd_dataset = "inlier"
    evaluate.ood_eval = True

    # msma
    config.msma = msma = ml_collections.ConfigDict()
    msma.min_timestep = 0.01  # Ignore first x% of sigmas
    msma.n_timesteps = 10  # Number of discrete timesteps to evaluate

    # data
    config.data = data = ml_collections.ConfigDict()
    data.dataset = "BRAIN"
    data.image_size = (168, 200, 152)  # For generating images
    data.uniform_dequantization = False
    data.centered = False
    data.num_channels = 2
    data.dir_path = "/DATA/Users/amahmood/braintyp/processed/"
    data.splits_path = "/home/braintypicality/dataset/"
    data.tumor_dir_path = "/DATA/Users/amahmood/tumor/"

    # model
    config.model = model = ml_collections.ConfigDict()
    model.sigma_max = 5000  # TODO: Do this for brain ds!
    model.sigma_min = 0.01
    model.num_scales = 1000
    model.beta_min = 0.1
    model.beta_max = 20.0
    model.dropout = 0.0
    model.embedding_type = "fourier"

    # optimization
    config.optim = optim = ml_collections.ConfigDict()
    optim.weight_decay = 0
    optim.optimizer = "Adam"
    optim.lr = 2e-4
    optim.beta1 = 0.9
    optim.eps = 1e-8
    optim.warmup = 5000
    optim.grad_clip = 1.0

    config.seed = 42
    config.device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )

    # Configuration for Hyperparam sweeps
    config.sweep = sweep = ml_collections.ConfigDict()
    param_dict = dict(
        optim_weight_decay={"distribution": "log_uniform", "min": 1e-6, "max": 1e-1},
        optim_optimizer={"values": ["Adam", "Adamax", "AdamW"]},
        optim_lr={
            "distribution": "log_uniform",
            "min": math.log(1e-5),
            "max": math.log(1e-2),
        },
        optim_beta1={"distribution": "uniform", "min": 0.9, "max": 0.999},
        optim_warmup={"values": [1000, 5000]},
        training_n_iters={"value": 1001},
        training_log_freq={"value": 50},
        training_eval_freq={"value": 100},
        training_snapshot_freq={"value": 1000},
        training_snapshot_freq_for_preemption={"value": 10000},
    )

    sweep.parameters = param_dict
    sweep.method = "random"
    sweep.metric = dict(name="val_loss")
    return config
