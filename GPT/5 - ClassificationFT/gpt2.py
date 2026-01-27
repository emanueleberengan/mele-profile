import torch
import tiktoken
from torch import nn
from torch.utils.data import Dataset, DataLoader

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

def generate_text_simple(model, idx, max_new_tokens, context_size):
    """
    Greedy generation (argmax) from an untrained model.
    idx: LongTensor of shape (B, T)
    """
    model.eval()
    for _ in range(max_new_tokens):
        # Keep only the last 'context_size' tokens
        idx_cond = idx[:, -context_size:]

        # Forward pass
        logits = model(idx_cond)  # (B, T, vocab_size)

        # Focus on the last time-step
        logits_last = logits[:, -1, :]  # (B, vocab_size)

        # Convert to probabilities and pick argmax
        probs = torch.softmax(logits_last, dim=-1)
        idx_next = torch.argmax(probs, dim=-1, keepdim=True)  # (B, 1)

        # Append to sequence
        idx = torch.cat((idx, idx_next), dim=1)
    return idx        


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )
        
    def forward(self, x):
        return self.layers(x)


class MultiHeadAttention(nn.Module):
    def __init__ (self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
               
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length,context_length),diagonal=1))
        
    def forward(self,x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        
        keys = keys.view(b,num_tokens,self.num_heads,self.head_dim)
        values = values.view(b,num_tokens,self.num_heads,self.head_dim)
        queries = queries.view(b,num_tokens,self.num_heads,self.head_dim)
        
        keys = keys.transpose(1,2)
        queries = queries.transpose(1,2)
        values = values.transpose(1,2)
        
        att_score = queries @ keys.transpose(2,3)
        mask_bool = self.mask.bool()[:num_tokens,:num_tokens]
        att_score.masked_fill_(mask_bool, -torch.inf)
        
        att_weights = torch.softmax(att_score / keys.shape[-1]**0.5,dim=-1)
        att_weights = self.dropout(att_weights)
        
        context_vec = (att_weights @ values).transpose(1,2)
        
        context_vec = context_vec.contiguous().view(b,num_tokens,self.d_out)
        context_vec = self.out_proj(context_vec)
        
        return context_vec
        

class LayerNorm(nn.Module):
    def __init__(self,emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
        
    def forward(self,x):
        mean = x.mean(dim=-1,keepdim=True)
        var = x.var(dim=-1,keepdim=True,unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
    

class TransformerBlock(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])
        
    def forward(self,x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        
        return x

class GPTModel(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.token_emb = nn.Embedding(cfg["vocab_size"],cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"],cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"],
            cfg["vocab_size"],
            bias=False
        )
        

    def forward(self,in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.token_emb(in_idx)
        pos_embeds = self.pos_emb(
            torch.arange(seq_len,device=in_idx.device)
        )
        x = tok_embeds + pos_embeds # sum the token emb with positional embeddings to get out informations
                                    # about the relevance of the single token in respect of it's position
        x = self.drop_emb(x)        # drop out to let the model generalize on unseen training data. Drop is applied only on training and not on inference 
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
        
        
############ Dataloaders 

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):

    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader

# a basic calculation of loss using the cross entropy function from torch
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0,1), target_batch.flatten() # Cross entropy fucntion applies to flatten batches
    )
    return loss

def calc_loss_dataloaders(data_loader, model, device, num_batches=None):
    total_loss = 0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min (num_batches, len(data_loader))
    
    for i, (input_batch, target_batch) in enumerate (data_loader):
        if i < num_batches:
            loss = calc_loss_batch(
                input_batch=input_batch,target_batch=target_batch,model=model,device=device
            )
            total_loss += loss.item()
        else:
            break
    return total_loss/num_batches

def text_to_token_ids(text,tokenizer):
    encoded = tokenizer.encode(text,allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def token_ids_to_text(token_ids,tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

def evaluate_model( model, train_loader, test_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_dataloaders(
            train_loader,model,device, num_batches=eval_iter
        )
        test_loss = calc_loss_dataloaders(
            test_loader,model,device, num_batches=eval_iter
        )
    model.train()
    return train_loss, test_loss

def generate_and_print_samples( model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context,tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(model=model,idx=encoded,max_new_tokens=50,context_size=context_size)
    decoded_text = token_ids_to_text(token_ids,tokenizer)
    print("decoded text: ", decoded_text.replace("\n"," "))

def train_model_simple(model, train_loader, test_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    train_losses, test_losses, track_token_seen = [], [], []
    token_seen, global_step = 0, -1
    
    # epoch cycle
    for epoch in range(num_epochs):
        model.train() # put the model in training mode for the backpropagation
        # and start cycling over input/target batches from the training dataloader
        for input_batch, target_batch in train_loader:
            
            optimizer.zero_grad() # at the beginning of each batch it's suggested to nullify the gradients and restart from scratch
            
            # get loss for the current batch
            loss = calc_loss_batch(
                input_batch,target_batch,model, device
                
            )
            
            loss.backward()
            optimizer.step()
            token_seen += input_batch.numel() # + one more in the stack of seen tokens
            
            global_step += 1
            
            # Just to print out the current status of training for the model object we are working with
            if global_step % eval_freq == 0:
                train_loss, test_loss = evaluate_model(
                    model, train_loader, test_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                test_losses.append(test_loss)
                track_token_seen.append(token_seen)
                print   (
                    f"Ep {epoch+1} (Step: {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, "
                    f"Test loss {test_loss:.3f}"
                )
        
        # let's print out some text samples to check the aility of the model to generate coherent text        
        generate_and_print_samples(
            model, tokenizer, device, start_context
        )
    
    return train_losses, test_losses, track_token_seen


def generate_t_k(model,idx,max_new_tokens,context_size, temperature=0.0,top_k=None,eos_id=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:,-context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
            
        logits = logits[:,-1,:]
        if top_k is not None:
            top_logits, _ = torch.topk(logits,top_k)
            min_val = top_logits[:,-1]
            
            logits = torch.where(
                logits < min_val,
                torch.tensor(float('-inf')).to(logits.device),
                logits
            )
            
        if temperature > 0.0 :
            logits = logits / temperature
            probs = torch.softmax(logits,-1)
            idx_next = torch.multinomial(probs,num_samples=1)
        if idx_next == eos_id:
            break
        
        idx = torch.cat((idx,idx_next),-1)
        
    return idx

