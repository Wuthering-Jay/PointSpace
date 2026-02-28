weight = None  # path to model weight
resume = False  # whether to resume training process
evaluate = True  # evaluate after each epoch training process
test_only = False  # test process

seed = None  # train process will init a random seed and record
save_path = "exp/default"
num_worker = 16  # total worker in all gpu
batch_size = 16  # total batch size in all gpu (legacy, prefer batch_size_train)
batch_size_train = None  # total training batch size; if set, overrides batch_size
gradient_accumulation_steps = 1  # total steps to accumulate gradients for
batch_size_val = None  # auto adapt to bs 1 for each gpu
batch_size_test = None  # auto adapt to bs 1 for each gpu; >1 = fragments per forward in SemSegTester
epoch = 100  # total epoch, data loop = epoch // eval_epoch
eval_epoch = 100  # sche total eval & checkpoint epoch
clip_grad = None  # disable with None, enable with a float

sync_bn = False
enable_amp = False
amp_dtype = "float16"
find_unused_parameters = False

enable_wandb = True
wandb_project = "pointcept"  # custom your project name e.g. Sonata, PTv3
wandb_key = None  # wandb token, default is None. If None, login with `wandb login` in your terminal

mix_prob = 0
param_dicts = None  # example: param_dicts = [dict(keyword="block", lr_scale=0.1)]

# hook
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter", interval=10),
    dict(type="CacheCleaner"),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]

# Trainer
train = dict(type="DefaultTrainer")

# Writer (optional, for saving results in practical formats like LAS/PLY/PCD)
# writer = dict(type="LASWriter", save_dir="exp/default/las_output/", source_dir=None)
writer = None

# Tester
test = dict(type="SemSegTester", verbose=True)
