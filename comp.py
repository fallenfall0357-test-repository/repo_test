import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# read text & build vocab (+ special tokens)
# -------------------------
with open('./intro_transformer.txt','r',encoding='utf-8') as f:
    text = f.read()

base_chars = sorted(list(set(text)))
# set special token
PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
itos = [PAD, BOS, EOS] + base_chars
stoi = {ch:i for i,ch in enumerate(itos)}

pad_id = stoi[PAD]
bos_id = stoi[BOS]
eos_id = stoi[EOS]

def encode(s): return [stoi[c] for c in s]
def decode(ids): 
    return ''.join([itos[i] for i in ids if itos[i] not in (PAD, BOS, EOS)])

train_ids = torch.tensor(encode(text), dtype=torch.long)

# -------------------------
# fuction：get batch from continuous text（copy task）
# src: length = block_size
# tgt_in: [BOS] + src (length block_size+1)
# tgt_out: src + [EOS] (length block_size+1)
# -------------------------
def get_batch(train_ids, block_size, batch_size, device):
    max_start = len(train_ids) - block_size - 1
    ix = torch.randint(0, max_start, (batch_size,))
    src = torch.stack([train_ids[i:i+block_size] for i in ix])
    tgt = src.clone()
    tgt_in = torch.full((batch_size, block_size+1), pad_id, dtype=torch.long)
    tgt_out = torch.full((batch_size, block_size+1), pad_id, dtype=torch.long)
    tgt_in[:,0] = bos_id
    tgt_in[:,1:] = tgt
    tgt_out[:,:block_size] = tgt
    tgt_out[:,block_size] = eos_id
    return src.to(device), tgt_in.to(device), tgt_out.to(device)

# -------------------------
# min module：positional encoding（sinusoidal）
# -------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer('pe', pe)  # (max_len, d_model)

    def forward(self, x):
        # x: (B, T, C)
        return x + self.pe[:x.size(1), :].unsqueeze(0).to(x.device)

# -------------------------
# multi-head attn（cross-attn supported）
# mask: causal mask of shape (Tq, Tk) with True where allowed
# -------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, q, k, v, mask=None):
        # q: (B, Tq, C), k/v: (B, Tk, C)
        B, Tq, C = q.shape
        Tk = k.shape[1]
        q = self.q_proj(q).view(B, Tq, self.num_heads, self.d_head).transpose(1,2)  # (B, nh, Tq, d)
        k = self.k_proj(k).view(B, Tk, self.num_heads, self.d_head).transpose(1,2)   # (B, nh, Tk, d)
        v = self.v_proj(v).view(B, Tk, self.num_heads, self.d_head).transpose(1,2)   # (B, nh, Tk, d)

        att = (q @ k.transpose(-2,-1)) / math.sqrt(self.d_head)  # (B, nh, Tq, Tk)

        if mask is not None:
            # mask shape (Tq, Tk) -> expand to (1,1,Tq,Tk)
            if mask.dim() == 2:
                m = mask.unsqueeze(0).unsqueeze(0).to(att.device)
            else:
                m = mask.to(att.device)
            att = att.masked_fill(~m, float('-inf'))

        att = F.softmax(att, dim=-1)
        out = att @ v  # (B, nh, Tq, d)
        out = out.transpose(1,2).contiguous().view(B, Tq, C)  # (B, Tq, C)
        return self.o_proj(out)

# -------------------------
# simple feed forward
# -------------------------
class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model*4)
        self.fc2 = nn.Linear(d_model*4, d_model)
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

# -------------------------
# Encoder / Decoder Layer（minimum）
# -------------------------
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.sa = MultiHeadAttention(d_model, num_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model)

    def forward(self, x):
        # self-attention
        x = x + self.sa(self.ln1(x), self.ln1(x), self.ln1(x))
        # feed-forward
        x = x + self.ff(self.ln2(x))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ln3 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model)

    def forward(self, x, enc_out, tgt_mask=None):
        # masked self-attn
        x = x + self.self_attn(self.ln1(x), self.ln1(x), self.ln1(x), mask=tgt_mask)
        # cross-attn (queries from decoder, keys/vals from encoder)
        x = x + self.cross_attn(self.ln2(x), enc_out, enc_out)
        # ff
        x = x + self.ff(self.ln3(x))
        return x

# -------------------------
# Transformer Encoder-Decoder（minimum）
# -------------------------
class SimpleTransformerEncDec(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_heads=4, num_enc_layers=2, num_dec_layers=2, max_len=128):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)
        self.enc_layers = nn.ModuleList([EncoderLayer(d_model, num_heads) for _ in range(num_enc_layers)])
        self.dec_layers = nn.ModuleList([DecoderLayer(d_model, num_heads) for _ in range(num_dec_layers)])
        self.ln_final = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, vocab_size)

    def encode(self, src):
        x = self.tok_emb(src)         # (B, Ts, C)
        x = self.pos_enc(x)
        for layer in self.enc_layers:
            x = layer(x)
        return x

    def decode(self, tgt_in, enc_out):
        x = self.tok_emb(tgt_in)     # (B, Tt, C)
        x = self.pos_enc(x)
        Tt = x.size(1)
        # causal mask: True where allowed
        causal = torch.tril(torch.ones(Tt, Tt, dtype=torch.bool, device=x.device))
        for layer in self.dec_layers:
            x = layer(x, enc_out, tgt_mask=causal)
        return x

    def forward(self, src, tgt_in):
        enc_out = self.encode(src)
        dec_out = self.decode(tgt_in, enc_out)
        logits = self.proj(self.ln_final(dec_out))
        return logits

    @torch.no_grad()
    def generate(self, src, max_new_tokens=100, temperature=1.0, top_k=50,
                bos_id=1, eos_id=2):
        """
        src: (B, S) source sequence
        """
        B = src.size(0)
        # initial target sequence：only BOS included
        tgt_in = torch.full((B, 1), bos_id, dtype=torch.long, device=src.device)

        for _ in range(max_new_tokens):
            # forward: encoder + decoder
            logits = self.forward(src, tgt_in)   # (B, Tt, vocab_size)
            logits = logits[:, -1, :]            # taking last step predict (B, vocab_size)

            # Temperature
            logits = logits / temperature

            # Top-k select
            if top_k is not None:
                v, ix = torch.topk(logits, k=top_k, dim=-1)
                mask = logits < v[:, [-1]]
                logits[mask] = -float("Inf")

            # softmax + sampling
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B,1)

            # concat to target sequence
            tgt_in = torch.cat((tgt_in, idx_next), dim=1)

            # if generate to EOS, early break
            if (idx_next == eos_id).all():
                break

        return tgt_in

# -------------------------
# hyperparameter & model
# -------------------------
block_size = 256      # src length
batch_size = 32
d_model = 512
num_heads = 8
num_enc_layers = 6
num_dec_layers = 6
max_tgt_len = block_size + 1  # because we will prepose BOS at tgt_in

model = SimpleTransformerEncDec(len(itos), d_model=d_model, num_heads=num_heads,
                                num_enc_layers=num_enc_layers, num_dec_layers=num_dec_layers,
                                max_len=max(block_size, max_tgt_len)).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# -------------------------
# Training loop（Sample：copy task）
# -------------------------
steps = 5000
model.train()
pbar = tqdm(range(steps), desc="Training")
for step in pbar:
    src, tgt_in, tgt_out = get_batch(train_ids, block_size, batch_size, device)
    logits = model(src, tgt_in)  # (B, Tt, V)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt_out.view(-1), ignore_index=pad_id)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if step % 200 == 0:
        pbar.set_postfix({'loss': loss.item()})

# save weights
torch.save(model.state_dict(), "simple_transformer_encdec.pth")

# -------------------------
# generate sample
# -------------------------
model.eval()
with torch.no_grad():
    user_input = "attentionとは"
    src_ids = encode(user_input)
    src_example = torch.tensor([src_ids], dtype=torch.long, device=device)

    gen_ids = model.generate(
    src_example,
    max_new_tokens=200,
    temperature=0.8,
    top_k=50,
    bos_id=bos_id,
    eos_id=eos_id
    )

    print("SRC :", user_input)
    print("GEN :", decode(gen_ids[0].tolist()))
