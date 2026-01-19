import torch
import tiktoken
import torch
from GPTModel import GPTModel
from TransformerBlock import TransformerBlock
from GPTConfigs import GPT_CONFIG_124M

## Sample test to generate the next token from untrained model
def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens): 
        print("idx shape:",idx.shape, "\n idx: ", idx)
        idx_cond = idx[:, -context_size:]
        print("idx_cond shape:",idx_cond.shape, "\n idx: ", idx_cond)
        with torch.no_grad():
            logits = model(idx_cond)
            print("model logits shape: ",logits.shape, "\n model logits: ", logits)

        logits = logits[:, -1, :]
        print("logits resahped shape:",logits.shape, "\n logitsreshaped: ", logits)
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim = True)
        idx = torch.cat((idx, idx_next), dim=1)
        print("idx shape after cat:",idx.shape, "\n idx: ", idx)
        print("\n\n##########\n\n")
    
    return idx
        


torch.manual_seed(254)
model = GPTModel(GPT_CONFIG_124M)

# Load tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# Text samples
text1 = "Every effort moves you"
text2 = "Every day holds a"

# Encode texts and build batch list
batch = []
batch.append(torch.tensor(tokenizer.encode(text1)))
batch.append(torch.tensor(tokenizer.encode(text2)))
batch = torch.stack(batch,dim=0)

out = model(batch)

print("Input batch:\n",batch)
print("Output out, shape:",out.shape,"\nOutput:\n",out)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}")

model.token_emb.weight.shape

start_context = "Hello, I am"
encoded = tokenizer.encode(start_context)
print ("Encoded:",  encoded)
encoded_tensor = torch.tensor(encoded).unsqueeze(0)
print("Encoded tensor shape:",encoded_tensor.shape)

model.eval()

out = generate_text_simple(model=model,idx=encoded_tensor,max_new_tokens=6,context_size=GPT_CONFIG_124M["context_length"])

print("out: ",out, "\n out len: ",len(out[0]))

decoded = tokenizer.decode(out.squeeze(0).tolist())
print("Decoded text: ",decoded)

