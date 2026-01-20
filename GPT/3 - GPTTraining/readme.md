
## 🍐 vs 🍎 — Simple GPT Training on a Tiny Dataset

The notebooks contained in this section demonstrates how to train a GPT‑style language model **from scratch** using a very small text corpus.  

To get the most on this content follow the order of Jupiter books:
 - 🥇[intro-training.ipynb](intro-training.ipynb): Text to tokens and back, generate from randoms and loss calculation 
 - 🥈[training.ipynb](training.ipynb): Train the model on a simple dataset

The workflow includes:

- loading and tokenizing text with the GPT‑2 tokenizer  
- splitting data into 90% training / 10% test  
- preparing dataloaders for next‑token prediction  
- implementing cross‑entropy loss over token batches  
- running a minimal PyTorch training loop (AdamW optimizer)  
- periodically evaluating training vs test loss  
- generating short text samples after each epoch  

Because the dataset is intentionally tiny, the model quickly **overfits**:  
  
  > training loss continues improving while test loss plateaus around epoch 3.  
  > This provides a clear illustration of the generalization limits of GPT‑style models when trained on insufficient data.

