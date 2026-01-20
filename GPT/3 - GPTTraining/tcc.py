import os, tiktoken, torch
from gpt2 import GPTModel,create_dataloader_v1, GPT_CONFIG_124M, generate_text_simple

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




