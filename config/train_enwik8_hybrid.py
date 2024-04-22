# train a miniature character-level enwik8 model. Based on tiny shakespeare

out_dir = 'out-enwik8-char-hybrid'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 100 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False
wandb_project = 'enwik8-char'
wandb_run_name = 'hybrid-char-level-lm'

dataset = 'enwik8'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 512 # context of up to 256 previous characters

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

# hybrid self-attention
hybrid = True
local_window_size = 100 # how many tokens for local attention
local_weight = 0.5 # weight of local attention

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 100000 # 5000
lr_decay_iters = 100000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially