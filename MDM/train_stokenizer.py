"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=2 train.py --compile=False --batch_size=16

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

"""
# Use default config.json
python train_stokenizer.py

# Use custom config file
python train_stokenizer.py --config config.json
"""

import os
import time
import math
import pickle
from contextlib import nullcontext
import argparse
import json


import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import networkx as nx
import re 

from model import GPTConfig, GPT
import matplotlib.pyplot as plt
from logger import get_logger
from stokenizer import STokenizer


# -----------------------------------------------------------------------------
# the input parameters

parser = argparse.ArgumentParser(description='Training of the NanoGPT.')

parser.add_argument('--config', type=str, default='config.json', help='Path to config.json file (default: config.json)')
parser.add_argument('--n_layer', type=int, default=None, help='Number of layers (override config.json if set)')  
parser.add_argument('--n_head', type=int, default=None, help='Number of attention heads (override config.json if set)')  
parser.add_argument('--n_embd', type=int, default=None, help='Size of the embeddings (override config.json if set)')
parser.add_argument('--max_iters', type=int, default=300000, help='Number of Iterations (default: 100000)')
#ADDED
parser.add_argument('--batch_size', type=int, default=32, help='Batch size (default: 32)')
parser.add_argument('--compile', type=lambda x: x.lower() in ('true', '1', 'yes'), default=True, help='Use PyTorch 2.0 compilation (default: True)')
parser.add_argument('--eval_iters', type=int, default=None, help='Number of eval iterations (default: max_iters//10000)')
parser.add_argument('--grad_accumulation_steps', type=int, default=1, help='Gradient accumulation steps (default: 1)')
parser.add_argument('--block_size', type=int, default=None, help='Override context length if set; otherwise use value from meta.pkl')

path_nb_pergraph = 5  # Number of paths to sample per graph in each batch
args = parser.parse_args()





# Load config from JSON file
config_path = args.config
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        json_config = json.load(f)
    print(f"Loaded configuration from {config_path}")
else:
    print(f"ERROR: Config file '{config_path}' not found!")
    print(f"Please ensure the config file exists at the specified path.")
    print(f"Current working directory: {os.getcwd()}")
    exit(1)





# Use command line args if provided, otherwise use config.json values, otherwise use defaults
n_layer = args.n_layer if args.n_layer is not None else json_config.get('n_layer', 12)
n_head = args.n_head if args.n_head is not None else json_config.get('n_head', 12)
n_embd = args.n_embd if args.n_embd is not None else json_config.get('n_embd', 768)
max_iters = args.max_iters
#ADDED
batch_size = args.batch_size
train_batch_size = batch_size
val_batch_size = batch_size
compile = args.compile

seed = 444

# Load additional config parameters from JSON
attn_pdrop = json_config.get('attn_pdrop', 0.1)
embd_pdrop = json_config.get('embd_pdrop', 0.1)
resid_pdrop = json_config.get('resid_pdrop', 0.1)
bos_token_id = json_config.get('bos_token_id', 35)
eos_token_id = json_config.get('eos_token_id', 0)
initializer_range = json_config.get('initializer_range', 0.02)
layer_norm_epsilon = json_config.get('layer_norm_epsilon', 1e-05)
n_ctx = json_config.get('n_ctx', 1024)
n_positions = json_config.get('n_positions', 1024)
n_inner = json_config.get('n_inner', None)
activation_function = json_config.get('activation_function', 'gelu_new')
vocab_size_from_config = json_config.get('vocab_size', None)



data_dir = f'data'
with open('meta.pkl', 'rb') as f:
    meta = pickle.load(f)
    
_meta_block_size = 200 #meta['block_size']
block_size = args.block_size if args.block_size is not None else _meta_block_size
#vocab_size = meta['vocab_size']

out_dir = f'out/{n_layer}_{n_head}_{n_embd}_seed{seed}'

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
eval_interval = max(1, max_iters // 50)
log_interval = max(1, max_iters // 200)
eval_iters = args.eval_iters if args.eval_iters is not None else max(10, max_iters // 10000)

eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
#dataset = 'reasoning'
gradient_accumulation_steps = args.grad_accumulation_steps # used to simulate larger batch sizes
# train_batch_size and val_batch_size are set from command line args above
# Don't override them here
#block_size = 64
# model
#n_layer = 1 #12
#n_head = 1 #12
#n_embd = 384 #768


dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 5e-4 # max learning rate 
#max_iters = 50000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = max_iters//20 # how many steps to warm up for
lr_decay_iters = max_iters # should be ~= max_iters per Chinchilla
min_lr = learning_rate/10 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
# keep the value from CLI; do not override here

'''check_type = 'shortest'
max_path_len = 10
max_new_tokens = 200
flag = 0
test_interval = 100'''
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
#exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # Ensure gradient accumulation steps is at least as large as the number of GPUs
    if gradient_accumulation_steps < torch.cuda.device_count():
        gradient_accumulation_steps = torch.cuda.device_count()
    assert gradient_accumulation_steps % torch.cuda.device_count() == 0
    gradient_accumulation_steps //= torch.cuda.device_count()
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
###torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)



# poor man's data loader
#train_data = np.memmap('train.bin', dtype=np.uint16, mode='r')
#val_data = np.memmap('train.bin', dtype=np.uint16, mode='r')



#### CHANGE DATA HERE
# Load data from prosqa_train_processed.json
import json
import random
#from transformers import AutoTokenizer

#LAST MODEL trained
#data_path = '../../data/prosqa_train_processed.json'

#data_path = '../../data/prosqa_train_processed.json'
#with open(data_path, 'r') as f:
 #   all_data = json.load(f)

# Load special tokens from nodes_names.json
#nodes_names_path = '../../data/nodes_names.json'
#with open(nodes_names_path, 'r') as f:
 #   special_node_names = json.load(f)

dataset_path= "/home/v-lberrayana/coconut/data/theoritical_paper/prosqa_train_graph_4_coconut.json"
with open(dataset_path, 'r') as f:
   all_data = json.load(f)
train_data = all_data
dataset_path= "/home/v-lberrayana/coconut/data/theoritical_paper/prosqa_valid_graph_4_coconut.json"
with open(dataset_path, 'r') as f:
   all_data = json.load(f)
val_data = all_data


# Initialize tokenizer
tokenizer = STokenizer()

#start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
#end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")


# Tokenize special node names to get their token IDs

#special_token_ids = set()
#for name in special_node_names:
  #  tokens = tokenizer.encode(name, add_special_tokens=False)
    #special_token_ids.update(tokens)
#special_token_ids = torch.tensor(list(special_token_ids), dtype=torch.int64)



print(f"Loaded {len(all_data)} samples: {len(train_data)} training, {len(val_data)} validation")


base_dataset = json.load(open(dataset_path))
    


def process_dataset(sample):
        # Shuffle edges for robustness
    random.shuffle(sample['edges'])
        
        # Construct the question part
    question = "<bos> " + "|".join([f" {e[0]} {e[1]} " for e in sample['edges']]).strip() + " [Q] "
        
        # Randomly order target and neg_target in the question
    if random.random() < 0.5:
        question += f"{sample['target']} {sample['neg_target']}"
    else:
        question += f"{sample['neg_target']} {sample['target']}"
            
    question += f" [R] {sample['root']}"
        
        # Construct the chain of thought (optimal path) and answer
    current_node = sample['root']
    continuation = ""
    for i in range(1, 10):
        if str(i) in sample["neighbor_k"]:
            next_node = random.choice(
                [n for n in sample["neighbor_k"][str(i)] if [current_node, n] in sample['edges']]
            )
            continuation += f" {next_node}"
            current_node = next_node
        
    continuation += f" [A] {sample['target']} <eos>"
    print(continuation)
   
        
        # Tokenize question and continuation
    question_tokenized = tokenizer.encode(question, add_special_tokens=False)
    continuation_tokenized = tokenizer.encode(continuation, add_special_tokens=False)
        
    tokens = question_tokenized + continuation_tokenized
        
    processed_sample = {
        "input_ids": tokens,
        "labels": [-100] * len(question_tokenized) + continuation_tokenized,
        "attention_mask": [1] * len(tokens),
        "position_ids": list(range(len(tokens))),
    }
         
    return question_tokenized, continuation_tokenized


'''train_data_split_points = []
for i in range(len(train_data)):
    if(train_data[i] == stoi["("]):
        train_data_split_points.append(i)
train_data_split_points.append(len(train_data))

val_data_split_points = []
for i in range(len(val_data)):
    if(val_data[i] == stoi["("]):
        val_data_split_points.append(i)
val_data_split_points.append(len(val_data))'''

def get_batch(split):
    data = train_data if split == 'train' else val_data
    batch_size = train_batch_size if split == 'train' else val_batch_size

    # Randomly sample batch_size samples from the data
    indices = torch.randint(len(data), (batch_size,))
    
    batch_x = []
    batch_y = []
    
    for idx in indices:
        for _ in range(path_nb_pergraph): # add multiple paths per graph
            sample = data[idx.item()]
            
            # Use process_dataset to get the processed sample
            question_tokens, answer_tokens = process_dataset(sample)

            
            tokens = question_tokens + answer_tokens
            question_len = len(question_tokens)
            answer_len = len(tokens) - question_len
            
            # Store original sequence length before padding
            #original_length = len(tokens)

            # Pad or truncate to block_size
            if len(tokens) < block_size:
                tokens = tokens + [pad_token_id] * (block_size - len(tokens)) #We add PAD TOKENS
            else:
                tokens = tokens[:block_size]
            
            tokens = torch.tensor(tokens, dtype=torch.int64)
            
            # x is the full sequence (question + answer)
            # y is the same, but we'll mask out the question part in the loss
            x = tokens.clone()
            y = tokens.clone()

            # Set y to -100 for question tokens (so they don't contribute to loss)
            y[:question_len] = -100
            
            
            # Mask random answer tokens for the model to predict
            if answer_len > 0:
                a = 1  # Single sample
                
                # Apply mask to answer tokens only
                answer_start = question_len
                answer_end = block_size # min(question_len + answer_len, block_size) : SHOULD LEARN TO GENERATE PAD TOKENS TOO !!!!
                x_answer = x[answer_start:answer_end].clone()
                
                # Calculate actual answer length (may be truncated)
                actual_answer_len = len(x_answer)
                
                if actual_answer_len > 0:
                    bp = actual_answer_len
                    
                    # Create probability weights favoring final 20 tokens
                    probs = torch.ones(bp)
                    
                    # Increase probability for final 20 tokens (2x higher)
                    #final_tokens_start = max(0, bp - 20)
                    #probs[final_tokens_start:] *= 100.0
                    
                    # Normalize probabilities
                    probs = probs / probs.sum()

                    k = torch.randint(1, bp+1, (1,)).item()
                    
                    # Sample positions to mask based on weighted probabilities
                    perm = torch.multinomial(probs.repeat(a, 1), bp, replacement=False)

                    
                    #print(perm)
                    #print(xx)
                    
                    mask = torch.zeros((a, bp), dtype=torch.bool)
                    mask.scatter_(1, perm[:, :k], True)
                    
                    x_answer[mask[0]] = mask_token_id # set to mask token id
                
                x[answer_start:answer_end] = x_answer
                
                # Update y to set non-masked positions to -100
                masked = (x == mask_token_id)
                y = torch.where(masked, y, -100)
            batch_x.append(x)
            batch_y.append(y)
    
    x_batch = torch.stack(batch_x)
    y_batch = torch.stack(batch_y)
    

    """
    .pin_memory() : Allocates the tensor in pinned (page-locked) memory on the CPU. This prevents the operating system from paging this memory to disk, which enables faster data transfers to the GPU.
    non_blocking=True : Allows the CPU-to-GPU transfer to happen asynchronously. This means:

    The CPU can continue preparing the next batch while the current batch is being transferred
    The GPU can start computing on already-transferred data while more data is still being moved
    This overlapping of data transfer and computation improves overall throughput
    """

    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x_batch, y_batch = x_batch.pin_memory().to(device, non_blocking=True), y_batch.pin_memory().to(device, non_blocking=True)
    else:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
    return x_batch, y_batch



# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# logger
logger = get_logger(os.path.join(out_dir, "train.log"))



pad_token_id = tokenizer.vocab[tokenizer.pad_token]
mask_token_id = tokenizer.vocab[tokenizer.mask_token]

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, 
                  vocab_size=vocab_size_from_config if vocab_size_from_config is not None else tokenizer.vocab_size,
                  mask_token_id=mask_token_id,
                  dropout=dropout,
                  # Additional config parameters
                  attn_pdrop=attn_pdrop,
                  embd_pdrop=embd_pdrop,
                  resid_pdrop=resid_pdrop,
                  bos_token_id=bos_token_id,
                  eos_token_id=eos_token_id,
                  initializer_range=initializer_range,
                  layer_norm_epsilon=layer_norm_epsilon,
                  n_ctx=n_ctx,
                  n_positions=n_positions,
                  n_inner=n_inner,
                  activation_function=activation_function) # start with model_args from command line



if init_from == 'scratch':
    print("Initializing a new model from scratch")
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])


torch.manual_seed(seed)

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item() 
        out[split] = losses.mean()
    model.train()
    # Evaluates model on train and validation splits
    # Returns average loss over multiple batches
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
accuracy = []
corrects = []
totals = []
while True:
    # refresh batch at the start of each iteration to account for any dynamic batch size changes
    X, Y = get_batch('train')
    
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        logger.info(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                logger.info(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, f'{iter_num}_ckpt.pt'))

    # if iter_num % test_interval == 0 and master_process:
    #     correct, tot = test_model()
    #     corrects.append(correct)
    #     totals.append(tot)

    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    try:
        for micro_step in range(gradient_accumulation_steps):
            if ddp:
                model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
            with ctx:
                logits, loss = model(X, Y)
                loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
            # prepare next micro-batch
            X, Y = get_batch('train')
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()

    except RuntimeError as e:
        # handle CUDA OOM by reducing batch size and retrying iteration (single GPU only)
        if 'out of memory' in str(e).lower():
            torch.cuda.empty_cache()
            if ddp:
                # In DDP, changing batch size on a single rank causes collective mismatches and NCCL timeouts.
                msg = (
                    "CUDA OOM detected in DDP. Dynamic batch shrink is disabled to avoid desync. "
                    "Please restart with smaller --batch_size and/or --block_size, or increase --grad_accumulation_steps."
                )
                print(msg)
                logger.info(msg)
                # re-raise to fail fast instead of hanging
                raise
            if train_batch_size > 1:
                old_bs = train_batch_size
                train_batch_size = max(1, train_batch_size // 2)
                val_batch_size = min(val_batch_size, train_batch_size)
                print(f"CUDA OOM detected. Reducing batch size from {old_bs} to {train_batch_size} and retrying this iteration.")
                logger.info(f"CUDA OOM detected. Reducing batch size from {old_bs} to {train_batch_size} and retrying this iteration.")
                optimizer.zero_grad(set_to_none=True)
                # retry the iteration with smaller batch size
                continue
            else:
                # can't reduce further; re-raise
                raise
        else:
            raise
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        logger.info(f"iter {iter_num}: loss {lossf:.4f}")
    iter_num += 1
    local_iter_num += 1

    if iter_num > max_iters:
        break

torch.save(torch.tensor(corrects).cpu(), os.path.join(out_dir, f'corrects.pt'))
torch.save(torch.tensor(totals).cpu(), os.path.join(out_dir, f'totals.pt'))

if ddp:
    destroy_process_group()