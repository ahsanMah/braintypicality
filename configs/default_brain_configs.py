import ml_collections
import torch
import math


def get_default_configs():
    config = ml_collections.ConfigDict()
    # training
    config.training = training = ml_collections.ConfigDict()
    config.training.batch_size = 3
    training.n_iters = 250001
    training.snapshot_freq = 10000
    training.log_freq = 100
    training.eval_freq = 500
    ## store additional checkpoints for preemption in cloud computing environments
    training.snapshot_freq_for_preemption = 1000
    ## produce samples at each snapshot.
    training.snapshot_sampling = True
    training.sampling_freq = 20000
    training.likelihood_weighting = False
    training.continuous = True
    training.reduce_mean = False
    # Pretrain options
    training.load_pretrain = False
    training.pretrain_dir = "/path/to/weights/"
    training.use_fp16 = False

    # sampling
    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.n_steps_each = 1
    sampling.noise_removal = True
    sampling.probability_flow = False
    sampling.snr = 0.075

    # evaluation
    config.eval = evaluate = ml_collections.ConfigDict()
    evaluate.begin_ckpt = 10
    evaluate.end_ckpt = 10
    evaluate.batch_size = 32
    evaluate.enable_sampling = True
    evaluate.num_samples = 8
    evaluate.enable_loss = False
    evaluate.enable_bpd = False
    evaluate.bpd_dataset = "inlier"
    evaluate.ood_eval = False
    evaluate.sample_size = 32

    # msma
    config.msma = msma = ml_collections.ConfigDict()
    msma.max_timestep = 1.0
    msma.min_timestep = 1e-3  # Ignore first x% of sigmas
    msma.n_timesteps = 20  # Number of discrete timesteps to evaluate
    msma.seq = "linear"  # Timestep schedule that dictates which sigma to sample
    msma.checkpoint = -1  # ckpt number for score norms, defaults to latest (-1)
    msma.skip_inliers = False  # skip computing score norms for inliers
    msma.apply_masks = False
    msma.expectation_iters = -1
    msma.denoise = False

    # data
    config.data = data = ml_collections.ConfigDict()
    data.dataset = "BRAIN"
    data.gen_ood = False
    data.ood_ds = "DS-SA"  # "IBIS"
    # data.image_size = (192, 224, 160)  # For generating images
    data.image_size = (176, 208, 160)
    data.spacing_pix_dim = 1.0
    data.uniform_dequantization = False
    data.centered = False
    data.dir_path = "/DATA/Users/amahmood/braintyp/processed_v2/"
    data.splits_path = "/home/braintypicality/dataset/"
    data.tumor_dir_path = "/DATA/Users/amahmood/tumor/"
    data.colab_path = "/content/drive/MyDrive/ML_Datasets/ABCD/processed/"
    data.colab_splits_path = "/content/drive/MyDrive/Developer/braintypicality/dataset/"
    data.colab_tumor_path = "/content/drive/MyDrive/ML_Datasets/ABCD/tumor/"
    data.as_tfds = False
    data.cache_rate = 0.0
    data.num_channels = 2
    data.select_channel = -1  # -1 = all, o/w indexed from zero

    # model
    config.model = model = ml_collections.ConfigDict()
    model.sigma_max = 545.0  # For medres
    model.sigma_min = 0.06
    model.num_scales = 1000
    model.beta_min = 0.1
    model.beta_max = 20.0
    model.dropout = 0.0
    model.embedding_type = "fourier"
    model.blocks_down = (1, 2, 2, 4)
    model.blocks_up = (1, 1, 1)
    model.resblock_pp = False
    model.dilation = 1
    model.jit = False
    model.resblock_type = "segresnet"
    model.self_attention = False

    # optimization
    config.optim = optim = ml_collections.ConfigDict()
    optim.weight_decay = 0.0
    optim.optimizer = "Adam"
    optim.scheduler = "skip"
    optim.lr = 3e-4
    optim.beta1 = 0.9
    optim.eps = 1e-8
    optim.warmup = 5000
    optim.grad_clip = 1.0
    optim.adaptive_loss = False

    config.seed = 42
    config.device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    config.colab = False

    # Configuration for Hyperparam sweeps
    config.sweep = sweep = ml_collections.ConfigDict()
    param_dict = dict(
        optim_weight_decay={
            "distribution": "log_uniform",
            "min": math.log(1e-6),
            "max": math.log(1e-1),
        },
        optim_optimizer={"values": ["Adam", "Adamax", "AdamW"]},
        # optim_lr={
        #     "distribution": "log_uniform",
        #     "min": math.log(1e-5),
        #     "max": math.log(1e-2),
        # },
        model_time_embedding_sz={"values": [128, 256]},
        model_attention_heads={"values": [1, 0]},
        # model_embedding_type={"values": ["fourier", "positional"]},
        optim_warmup={"values": [5000]},
        optim_scheduler={"values": ["skip"]},
        training_n_iters={"value": 50001},
        training_log_freq={"value": 50},
        training_eval_freq={"value": 100},
        training_snapshot_freq={"value": 100000},
        training_snapshot_freq_for_preemption={"value": 100000},
    )

    sweep.parameters = param_dict
    sweep.method = "bayes"
    sweep.metric = dict(name="val_loss")
    return config
