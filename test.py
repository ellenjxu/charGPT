"""
Evaluates trained model on test set (bpc).
"""

import os
import math
import numpy as np
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file

dataset = 'enwik8'
eval_iters = 200
block_size = 512
batch_size = 12
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

data_dir = os.path.join('data', dataset)
def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    elif split == 'test':
        data = np.memmap(os.path.join(data_dir, 'test.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_test_loss():
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch('test')
        with ctx:
            logits, loss = model(X, Y)
        losses[k] = loss.item()
    out = losses.mean()
    model.train()
    return out

# def get_batch_iterator(split):
#     """splits the test data into batches with block size max sequence lengths"""
#     data_path = os.path.join(data_dir, f'{split}.bin')
#     data = np.memmap(data_path, dtype=np.uint16, mode='r')
#     num_batches = len(data) // (batch_size * block_size) # non-overlapping batches
    
#     for batch_idx in range(num_batches):
#         # process batch_size*block_size tokens in parallel
#         start_idx = batch_idx * batch_size * block_size
#         end_idx = start_idx + batch_size * block_size
#         batch_data = data[start_idx:end_idx]
#         print(batch_data.shape)
        
#         x = torch.stack([torch.from_numpy(batch_data[i*block_size:(i+1)*block_size].astype(np.int64)) for i in range(batch_size)])
#         y = torch.stack([torch.from_numpy(batch_data[i*block_size+1:(i+1)*block_size+1].astype(np.int64)) for i in range(batch_size)])
#         print(f"batch {batch_idx} shape: {x.shape}, {y.shape}")
        
#         if device_type == 'cuda':
#             x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
#         else:
#             x, y = x.to(device), y.to(device)
        
#         yield x, y

# @torch.no_grad()
# def estimate_bpc_loss(split):
#     """Estimate bpc loss for the test dataset"""
#     model.eval()
#     print(f"Estimating bpc loss for {split} set...")
#     batch_iter = get_batch_iterator(split)
#     total_loss = 0
#     total_batches = 0
    
#     for X, Y in batch_iter: # go through entire test set
#         print(X, Y)
#         with ctx:
#             _, loss = model(X, Y)
#         total_loss += loss.item()
#         total_batches += 1
#         print(f"Batch {total_batches} loss: {loss.item()}")
#         break
    
#     avg_loss = total_loss / total_batches
#     bpc = avg_loss / math.log(2)
#     print(f"{split} bpc: {bpc}")
#     return bpc
    
loss = estimate_test_loss()
bpc = loss / math.log(2)
print(f"loss: {loss}, bpc: {bpc}")
