import ml_collections
import torch


def get_default_configs():
    config = ml_collections.ConfigDict()
    # training
    config.training = training = ml_collections.ConfigDict()
    config.training.batch_size = 32
    training.n_iters = 2400001
    training.snapshot_freq = 5001
    training.log_freq = 50
    training.eval_freq = 100
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
    evaluate.batch_size = 16
    evaluate.enable_sampling = False
    evaluate.num_samples = 50000
    evaluate.enable_loss = True
    evaluate.enable_bpd = True
    evaluate.bpd_dataset = "inlier"
    evaluate.ood_eval = True

    # msma
    config.msma = msma = ml_collections.ConfigDict()
    msma.min_timestep = 0.1  # Ignore first 10% of sigmas
    msma.n_timesteps = 10  # Number of discrete timesteps to evaluate

    # data
    config.data = data = ml_collections.ConfigDict()
    data.dataset = "KNEE"
    data.image_size = 128  # Used by the model creator
    data.original_dimensions = (220, 160)
    data.downsample_size = 128
    data.random_flip = True
    data.uniform_dequantization = False
    data.centered = False
    data.dir_path = "/home/PO3D/raw_data/knee/"
    data.dir_path_longleaf = "/nas/longleaf/home/amahmood/mypine/DATA/knee"

    data.mask_marginals = True
    data.num_channels = 2
    data.shuffle_size = 1000

    # model
    config.model = model = ml_collections.ConfigDict()
    model.sigma_max = 700  # Avg of max-pairwise dist = 550, 95 quanitle=700
    model.sigma_min = 0.01
    model.num_scales = 2000
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
    config.longleaf = False  # Flag that indicates running on cluster

    return config
