"""
torchrun --standalone --nproc_per_node=2 train.py --compile=False --batch_size=16



This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext
import argparse


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

parser.add_argument('--n_layer', type=int, default=2, help='Number of layers (default: 1)')  
parser.add_argument('--n_head', type=int, default=8, help='Number of attention heads (default: 1)')  
parser.add_argument('--n_embd', type=int, default=768, help='Size of the embeddings (default: 384)')
parser.add_argument('--max_iters', type=int, default=10000000, help='Number of Iterations (default: 100000)')
#ADDED
parser.add_argument('--batch_size', type=int, default=32, help='Batch size (default: 32)')
parser.add_argument('--compile', type=lambda x: x.lower() in ('true', '1', 'yes'), default=True, help='Use PyTorch 2.0 compilation (default: True)')
parser.add_argument('--eval_iters', type=int, default=None, help='Number of eval iterations (default: max_iters//10000)')
parser.add_argument('--grad_accumulation_steps', type=int, default=1, help='Gradient accumulation steps (default: 1)')
parser.add_argument('--block_size', type=int, default=None, help='Override context length if set; otherwise use value from meta.pkl')




args = parser.parse_args()

n_layer = args.n_layer
n_head = args.n_head
n_embd = args.n_embd
max_iters = args.max_iters
#ADDED
batch_size = args.batch_size
train_batch_size = batch_size
val_batch_size = batch_size
compile = args.compile


seed = 113



data_dir = f'data'
#with open('meta.pkl', 'rb') as f:
#    meta = pickle.load(f)
    
#block_size = meta['block_size']
#vocab_size = meta['vocab_size'] - 1

_meta_block_size = 300 #meta['block_size']
block_size = args.block_size if args.block_size is not None else _meta_block_size


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
#gradient_accumulation_steps = 1 # used to simulate larger batch sizes
#train_batch_size = 1024 # if gradient_accumulation_steps > 1, this is the micro-batch size
#val_batch_size = 1024
#batch_size = train_batch_size
#block_size = 64
# model
#n_layer = 1 #12
#n_head = 1 #12
#n_embd = 384 #768


dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate =  0.0001 #5e-4 # max learning rate 
#max_iters = 50000 # total number of training iterations
weight_decay = 0.02 #1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = max_iters//20 # how many steps to warm up for
lr_decay_iters = max_iters # should be ~= max_iters per Chinchilla
min_lr = learning_rate # Constant LR/10 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'float16' #'bfloat16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
#compile = True # use PyTorch 2.0 to compile the model to be faster

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

import json
import random


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

def find_path_dfs(edges, root, target):
    # Build adjacency list
    adj = {}
    for u, v in edges:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)   # undirected

    visited = set()
    path = []

    def dfs(node):
        visited.add(node)
        path.append(node)

        if node == target:
            return True  # found the goal

        # DFS on neighbors
        for nxt in adj.get(node, []):
            if nxt not in visited:
                if dfs(nxt):
                    return True

        # Backtrack
        path.append(node)   # record backtrack step (e.g., 2 → 3 → 2)
        return False

    dfs(root)
    return path


def build_question_from_sample(sample, shuffle_edges=True):
    """
    Build question string from sample. Extracted as helper function to avoid duplication.
    """
    edges = sample['edges'].copy() if not shuffle_edges else sample['edges']
    if shuffle_edges:
        random.shuffle(edges)
    
    question = "<bos> " + "|".join([f" {e[0]} {e[1]} " for e in edges]).strip() + " [Q] "
    
    # Randomly order target and neg_target in the question
    if random.random() < 0.5:
        question += f"{sample['target']} {sample['neg_target']}"
    else:
        question += f"{sample['neg_target']} {sample['target']}"
    
    question += f" [R] {sample['root']}"
    return question


def process_dataset(sample):
    # Construct the question part using helper
    question = build_question_from_sample(sample, shuffle_edges=True)

    # ----------- NEW PART: DFS TREE SEARCH -----------
    path = find_path_dfs(
        edges=sample['edges'],
        root=sample['root'],
        target=sample['target']
    )

    # turn into continuation text
    continuation = ""
    for node in path[1:]:       # skip the root, already implied
        continuation += f" {node}"

    continuation += f" [A] {sample['target']} <eos>"
    print(continuation)
    # ---------------------------------------------------

    # Tokenization
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
        sample = data[idx.item()]
        
        # Use process_dataset to get the processed sample
        question_tokens, answer_tokens = process_dataset(sample)

        # build sequence and pad/truncate to block_size
        tokens = (question_tokens + answer_tokens)[:block_size]           # truncate first
        tokens = tokens + [pad_token_id] * max(0, block_size - len(tokens))
        tokens = torch.tensor(tokens, dtype=torch.int64)

        # number of question tokens after possible truncation
        question_len = min(len(question_tokens), len(tokens))

        # inputs are the tokens as-is
        x = tokens.clone()

        # targets are next-token predictions: shift left (y[i] = tokens[i+1]) !!!!!!
        # last position has no target -> -100
        y = torch.full_like(tokens, -100)
        if tokens.numel() >= 2:
            y[:-1] = tokens[1:]
            y[-1] = -100

        # mask out question tokens (they should not contribute to loss)
        y[:question_len-1] = -100

        # mask padding tokens
        y[tokens == pad_token_id] = -100

        # debug: ensure there is at least one label token to predict
        if (y != -100).sum().item() == 0:
            raise RuntimeError(f"No target tokens in sample idx={idx.item()} after truncation. question_len={question_len}, block_size={block_size}")

        batch_x.append(x)
        batch_y.append(y)


    
    x_batch = torch.stack(batch_x)
    y_batch = torch.stack(batch_y)
    
    
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x_batch, y_batch = x_batch.pin_memory().to(device, non_blocking=True), y_batch.pin_memory().to(device, non_blocking=True)
    else:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
    return x_batch, y_batch


# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9
best_val_accuracy = 0.0

# logger
logger = get_logger(os.path.join(out_dir, "train.log"))



pad_token_id = tokenizer.vocab[tokenizer.pad_token]


# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=tokenizer.vocab_size, dropout=dropout) # start with model_args from command line
if init_from == 'scratch':
    print("Initializing a new model from scratch")
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
# Note: checkpoint loading would go here if init_from == 'resume'
# if init_from == 'resume':
#     checkpoint = torch.load('path_to_checkpoint')
#     optimizer.load_state_dict(checkpoint['optimizer'])
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
    return out

# helps estimate accuracy on validation set using is_correct function
@torch.no_grad()
def estimate_accuracy(num_samples=None):
    """
    Estimate accuracy by generating predictions and comparing with ground truth.
    """
    if num_samples is None:
        num_samples = min(len(val_data), eval_iters * val_batch_size)
    
    model.eval()
    correct = 0
    total = 0
    
    # Sample validation examples
    indices = torch.randperm(len(val_data))[:num_samples]
    
    for idx in indices:
        sample = val_data[idx.item()]
        
        # Use helper function to build question (reusing logic from process_dataset)
        question = build_question_from_sample(sample, shuffle_edges=True)
        
        # Tokenize question
        question_tokens = tokenizer.encode(question, add_special_tokens=False)
        x = torch.tensor(question_tokens, dtype=torch.int64).unsqueeze(0).to(device)
        
        # Generate prediction (limit to block_size - len(question))
        max_new_tokens = min(50, block_size - len(question_tokens))
        
        with ctx:
            generated_ids = raw_model.generate(x, max_new_tokens=max_new_tokens, temperature=0.7, top_k=10, pad_token_id=pad_token_id)
        
        # Decode generated sequence
        generated_text = tokenizer.decode(generated_ids[0].tolist())
        
        # Get ground truth
        ground_truth = str(sample['target'])
        
        # Check if correct
        if is_correct(generated_text, ground_truth):
            correct += 1
        total += 1
    
    accuracy = correct / total if total > 0 else 0.0
    model.train()
    return accuracy

def is_correct(generated_answer, ground_truth):
    """
    Compare only the last sentence (after the final period) of both answers.
    Returns True if they match.
    """

    print("Generated answer:", generated_answer)
    print("Ground truth answer:", ground_truth)
        
    # Function to extract the last sentence after the last period
    def get_answer(sequence):
        if sequence is None:
            return ""
        if not isinstance(sequence, str):
            sequence = str(sequence)
        matches = re.findall(r'\[A\]\s*(.*)', sequence, flags=re.S)
        ans = matches[-1] if matches else ""
        # collapse all whitespace and strip edges
        return " ".join(ans.split()).strip()
    
    # Extract last sentences
    ground_truth_last = ground_truth
    generated_last = get_answer(generated_answer)

    print(f"DEEEEFFFF : Ground TRUTH: '{ground_truth_last}'")
    print(f"DEEEEFFFF : Generated : '{generated_last}'")

    if ground_truth_last == "" or generated_last == "":
        return False
    try:
        return int(ground_truth_last) == int(generated_last)
    except (ValueError, TypeError):
        # if either value cannot be parsed as int, treat as incorrect
        return False

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
    
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        val_accuracy = estimate_accuracy()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, val accuracy {val_accuracy:.4f}")
        logger.info(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, val accuracy {val_accuracy:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "val/accuracy": val_accuracy,
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })
        # Save checkpoint if accuracy improved (higher is better) or if always_save_checkpoint is True
        if val_accuracy > best_val_accuracy or always_save_checkpoint:
            best_val_accuracy = val_accuracy
            checkpoint = {
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'iter_num': iter_num,
                'best_val_loss': losses['val'],
                'best_val_accuracy': best_val_accuracy,
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
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
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