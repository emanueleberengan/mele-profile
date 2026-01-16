import torch
from torch import nn
from TransformerBlock import TransformerBlock, LayerNorm



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
        
        
        