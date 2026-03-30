"""
Deliverables generated automatically:
  • best_attn_glove_L{N}.pt          — best model checkpoint
  • training_curves_L{N}.png         — per-layer loss + F1 curves
"""
import math
import json, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

#constantts
bestmodelpath=r"C:\Users\Shivam Kumar\Downloads\bestmodelnlpq3.pt"

embedpathglove = r"C:\Users\Shivam Kumar\Downloads\glove_embeddings_200d.pkl"
trainpath = r"C:\Users\Shivam Kumar\Downloads\dataset\dataset\train_data.jsonl"
valpath   = r"C:\Users\Shivam Kumar\Downloads\dataset\dataset\val_data.jsonl"
embeddimenion=200
dimhiden = 256          # per BiLSTM direction  → total = 512
dropout  = 0.3
sizeofbatch= 32
epochs     = 3
learnrate  = 1e-3
grad_clip = 5.0
layerruns = [1, 2, 3] #given in assig
labeltoidmap = {"O": 0, "B-LOC": 1, "I-LOC": 2, "<PAD>": 3}
idtolabelmap = {v: k for k, v in labeltoidmap.items()}
padlabel= labeltoidmap["<PAD>"]
numberoflabels= 3        

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
#data loading
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]
#preprocesingg
#basically id for the label too 
def preprocess(data, vocab):
    samples = []
    for item in data:
        tokenid=[]
        labelid=[]
        for tok in item["tokens"]:
            tokenid.append(vocab.get(tok.lower(),vocab["<UNK>"]))
        for lab in item["labels"]:
            labelid.append(labeltoidmap.get(lab, labeltoidmap["O"]))
        samples.append((tokenid,labelid))
    return samples
#structure is like i have multiple data points
#it loops inside (currently x2 is used)
def build_vocab(datasets):
    #x1
    unique_tokens = {
        tok.lower() 
        for data in datasets 
        for item in data 
        for tok in item["tokens"]
    }
    vocabulary = {"<PAD>": 0, "<UNK>": 1}
    for i, token in enumerate(sorted(unique_tokens), start=2):
        vocabulary[token] = i
    
    #x2
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for data in datasets:
        for item in data:
            for tok in item["tokens"]:
                w = tok.lower()
                if w not in vocab:
                    vocab[w] = len(vocab)
    return vocab
#pretrained embeds
def load_glovepretrain(path, vocab, embed_dim=200):
    print("Loading GloVe pkl …")
    with open(path, "rb") as f:
        glove = pickle.load(f)          # dict {word: np.array(200,)}
    V = len(vocab)
    matrix = np.zeros((V, embed_dim), dtype=np.float32)  #dimension weill be vocab*200
    found = 0
    for word, idx in vocab.items():
        if word in glove:
            matrix[idx] = glove[word]
            found += 1
        elif idx > 1:                   # skip PAD and UNK placeholders
            matrix[idx] = np.random.uniform(-0.1, 0.1, embed_dim)
    # UNK = mean of all initialised vectors
    matrix[vocab["<UNK>"]] = matrix[2:].mean(axis=0)
    print(f"  GloVe coverage: {found}/{V} ({100*found/V:.1f}%)")
    return matrix

class NERDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]
#this prepares batches of different lenghts by padding when needed
from torch.nn.utils.rnn import pad_sequence
def collatefast(batch):
    tok_seqs, lbl_seqs = zip(*batch)
    tok_tensors = [torch.tensor(t) for t in tok_seqs]
    lbl_tensors = [torch.tensor(l) for l in lbl_seqs]
    lengths = torch.tensor([len(t) for t in tok_tensors])
    pad_tok = pad_sequence(tok_tensors, batch_first=True, padding_value=0)
    pad_lbl = pad_sequence(lbl_tensors, batch_first=True, padding_value=padlabel)
    return pad_tok, pad_lbl, lengths
def collate(batch):
    tok_seqs, lbl_seqs = zip(*batch)
    lengths   = [len(t) for t in tok_seqs]
    max_len   = max(lengths)
    B         = len(batch)
    pad_tok   = torch.zeros(B, max_len, dtype=torch.long)
    pad_lbl   = torch.full((B, max_len), padlabel, dtype=torch.long)
    for i, (t, l) in enumerate(zip(tok_seqs, lbl_seqs)):
        L = len(t)
        pad_tok[i, :L] = torch.tensor(t)
        pad_lbl[i, :L] = torch.tensor(l)
    return pad_tok, pad_lbl, torch.tensor(lengths)

########!!!!!!!!!!!
#bahdanau partttt
class bahdanauattention(nn.Module):
    """
    For each query position t:
        e_{t,s}  = v^T · tanh( W_q·h_t  +  W_k·h_s )
        α_{t,s}  = softmax_s( e_{t,s} )          (padded keys → -inf)
        context_t = Σ_s α_{t,s} · h_s

    Input  H : (B, T, D)
    Output   : context (B, T, D),  weights (B, T, T)
    """
    def __init__(self, dim):
        super().__init__()
        self.W_q = nn.Linear(dim, dim, bias=False)
        self.W_k = nn.Linear(dim, dim, bias=False)
        self.v   = nn.Linear(dim, 1,   bias=False)

    def forward(self, H, mask=None):
        # H : (B, T, D)
        Q = self.W_q(H).unsqueeze(2)          # (B, T, 1, D)
        K = self.W_k(H).unsqueeze(1)          # (B, 1, T, D)
        energy = self.v(torch.tanh(Q + K)).squeeze(-1)   # (B, T, T)

        if mask is not None:                   # mask: True = real token
            pad_mask = (~mask).unsqueeze(1)    # (B, 1, T) — padded keys
            energy = energy.masked_fill(pad_mask, float("-inf"))

        weights = torch.softmax(energy, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)   # all-pad rows → 0
        context = torch.bmm(weights, H)                # (B, T, D)
        return context, weights

class bilstmbahdanauneer(nn.Module):
    """
    Embedding → Stacked BiLSTM (L layers) → BahdanauAttention
              → concat[H, context] → Dropout → Linear → BIO logits
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers,
                 num_labels, dropout, pretrained=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained is not None:
            self.embedding.weight.data.copy_(torch.tensor(pretrained))

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        D = hidden_dim * 2                        # BiLSTM output dim
        self.attention  = bahdanauattention(D)
        self.drop       = nn.Dropout(dropout)
        self.classifier = nn.Linear(D * 2, num_labels)   # [H; ctx]

    def forward(self, tok_ids, lengths):
        mask  = (tok_ids != 0)                    # (B, T) real-token mask
        emb   = self.drop(self.embedding(tok_ids))
        packed = pack_padded_sequence(emb, lengths.cpu(),
                                      batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(packed)
        H, _        = pad_packed_sequence(lstm_out, batch_first=True)   # (B,T,D)

        context, _  = self.attention(H, mask=mask)
        out         = self.drop(torch.cat([H, context], dim=-1))       # (B,T,2D)
        return self.classifier(out)                                     # (B,T,C)

def extractspanentity(label_seq):
    #BIO → set of (start, end_inclusive, type) entity spans.
    spans, start = set(), None
    for i, lab in enumerate(label_seq):
        if lab.startswith("B-"):
            if start is not None:
                spans.add((start, i - 1, "LOC"))
            start = i
        elif lab.startswith("I-"):
            if start is None:
                start = i
        else:
            if start is not None:
                spans.add((start, i - 1, "LOC"))
                start = None
    if start is not None:
        spans.add((start, len(label_seq) - 1, "LOC"))
    return spans

def compute_metrics(all_preds, all_golds):
    #FreeMatch-F1 : entity-span level  (padded positions excluded)
    #Strict EM    : sentence-level exact label match (padded excluded)
    tp = fp = fn = 0
    em_correct = em_total = 0
    #firstly we remove any pad lables
    for preds, golds in zip(all_preds, all_golds):
        pairs    = [(p, g) for p, g in zip(preds, golds) if g != padlabel]
        if not pairs:
            continue
        p_lbls = [idtolabelmap[p] for p, _ in pairs]
        g_lbls = [idtolabelmap[g] for _, g in pairs]
        pred_spans = extractspanentity(p_lbls)
        gold_spans = extractspanentity(g_lbls)
        tp += len(pred_spans & gold_spans)
        fp += len(pred_spans - gold_spans)
        fn += len(gold_spans - pred_spans)

        em_total += 1
        if p_lbls == g_lbls:
            em_correct += 1
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    em   = em_correct / (em_total + 1e-9)
    return {"FreeMatch-F1": f1, "Strict-EM": em,"Precision": prec, "Recall": rec}
#train!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def run_epoch(model, loader, optimizer, criterion, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_preds, all_golds = [], []

    grad_ctx = torch.enable_grad() if train else torch.no_grad()
    with grad_ctx:
        for tok, lbl, lengths in loader:
            tok, lbl = tok.to(DEVICE), lbl.to(DEVICE)
            lengths  = lengths.to(DEVICE)

            logits = model(tok, lengths)          # (B, T, C)
            B, T, C = logits.shape
            loss = criterion(logits.view(B * T, C), lbl.view(B * T))

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(-1).cpu().tolist()
            golds = lbl.cpu().tolist()
            all_preds.extend(preds)
            all_golds.extend(golds)

    metrics = compute_metrics(all_preds, all_golds)
    return total_loss / len(loader), metrics
#main train loop!!!!!!!
#########
def train_one(num_layers, embed_matrix, vocab_size,train_loader, val_loader):
    print(f"\n{'━'*58}")
    print(f"bilstm (L={num_layers}) + bahdanau attention globe 200d")
    #creating a model with settings given
    model = bilstmbahdanauneer(vocab_size=vocab_size, embed_dim=embeddimenion,hidden_dim=dimhiden, num_layers=num_layers,num_labels=numberoflabels, dropout=dropout,pretrained=embed_matrix).to(DEVICE)
    #this counts trainable params
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {total_params:,}")
    criterion = nn.CrossEntropyLoss(ignore_index=padlabel)
    optimizer = optim.Adam(model.parameters(), lr=learnrate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)

    history = {k: [] for k in["train_loss", "val_loss", "train_f1", "val_f1","train_em",   "val_em"]}
    best_val_f1  = -1.0
    best_state   = None
    best_metrics = None

    for epoch in range(1, epochs + 1):
        tr_loss, tr_m = run_epoch(model, train_loader, optimizer, criterion, train=True)
        vl_loss, vl_m = run_epoch(model, val_loader,   optimizer, criterion, train=False)
        scheduler.step(vl_m["FreeMatch-F1"])

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_f1"].append(tr_m["FreeMatch-F1"])
        history["val_f1"].append(vl_m["FreeMatch-F1"])
        history["train_em"].append(tr_m["Strict-EM"])
        history["val_em"].append(vl_m["Strict-EM"])

        if vl_m["FreeMatch-F1"] > best_val_f1:
            best_val_f1  = vl_m["FreeMatch-F1"]
            best_metrics = vl_m
            best_state   = {k: v.cpu().clone()for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Ep {epoch:3d} | "
                  f"TrLoss {tr_loss:.4f}  TrF1 {tr_m['FreeMatch-F1']:.4f} | "
                  f"VlLoss {vl_loss:.4f}  VlF1 {vl_m['FreeMatch-F1']:.4f}  "
                  f"VlEM {vl_m['Strict-EM']:.4f}")
    ckpt_name = f"best_attn_glove_l{num_layers}.pt"
    torch.save({"num_layers":   num_layers,"vocab_size":   vocab_size,"embed_dim":    embeddimenion,"hidden_dim":   dimhiden,"num_labels":   numberoflabels,"dropout":      dropout,"state_dict":   best_state,"best_metrics": best_metrics,}, ckpt_name)
    print(f"\ncheckpoint location : {ckpt_name}")
    print(f"best val FreeMatch-F1 : {best_metrics['FreeMatch-F1']:.4f}")
    print(f"best val Strict-EM    : {best_metrics['Strict-EM']:.4f}")

    return history, best_metrics, ckpt_name
colors = {1: "#3b9fe6", 2: "#eb8f3e", 3: "#a0eaa0"}

def plot_single(l, hist, fname):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    ep = range(1, len(hist["train_loss"]) + 1)
    metrics = [("train_loss", "val_loss", "cross-entropy loss"),("train_f1",   "val_f1",   "freematch-f1"),("train_em",   "val_em",   "strict em"),]
    for ax, (tr_key, val_key, title) in zip(axes, metrics):
        ax.plot(ep, hist[tr_key], "--", color=colors[1], label="train")
        ax.plot(ep, hist[val_key], "-",  color=colors[1], label="val")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.legend()
        ax.grid(alpha=0.3)

    fig.suptitle(f"bilstm + bahdanau attention   l={l}  glove 200d", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()

#!!!!!!!!!!!!############
#for inference!!!!!!!
def load_model_from_ckpt(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model = bilstmbahdanauneer(vocab_size=ckpt["vocab_size"],embed_dim=ckpt["embed_dim"],hidden_dim=ckpt["hidden_dim"],num_layers=ckpt["num_layers"],num_labels=ckpt["num_labels"],dropout=ckpt["dropout"],).to(DEVICE)
    model.load_state_dict({k: v.to(DEVICE)for k, v in ckpt["state_dict"].items()})
    model.eval()
    return model, ckpt.get("best_metrics", {})
def predict(model, vocab, tokens):
    # tokens : list[str]   raw tokens of a single sentence
    # Returns: list[str]   predicted BIO labels
    ids = []
    for t in tokens:
        token = t.lower()
        if token in vocab:
            ids.append(vocab[token])
        else:
            ids.append(vocab["<UNK>"])
    tok_tensor = torch.tensor([ids], dtype=torch.long).to(DEVICE)
    lengths    = torch.tensor([len(ids)])
    with torch.no_grad():
        logits = model(tok_tensor, lengths)        # (1, T, C)
    pred_ids = logits.argmax(-1).squeeze(0).cpu().tolist()
    labels = []
    for p in pred_ids:
        labels.append(idtolabelmap[p])
    return labels

def main():
    #load preprocesss
    print("\nLoading data …")
    train_raw = load_jsonl(trainpath)
    val_raw   = load_jsonl(valpath)
    print(f"  Train sentences: {len(train_raw)}  |  Val: {len(val_raw)}")
    #vocabulary builder
    vocab = build_vocab([train_raw, val_raw])
    print(f"  Vocabulary size : {len(vocab)}")

    #i have a pretrained embed 
    embed_matrix = load_glovepretrain(embedpathglove, vocab, embeddimenion)
    #preprocess basically label id vocab id together
    train_samples = preprocess(train_raw, vocab)
    val_samples   = preprocess(val_raw,   vocab)
    train_loader = DataLoader(NERDataset(train_samples), batch_size=sizeofbatch,shuffle=True,  collate_fn=collate)
    val_loader   = DataLoader(NERDataset(val_samples),   batch_size=sizeofbatch,shuffle=False, collate_fn=collate)

    histories, results, checkpoints = [], [], []
    for L in layerruns:
        hist, metrics, ckpt = train_one(L, embed_matrix, len(vocab),train_loader, val_loader)
        histories.append(hist)
        results.append({"L": L, **metrics})
        checkpoints.append(ckpt)
        plot_single(L, hist, f"training_curves_L{L}.png")

    #plots
    print("\nGenerating deliverable plots …")

    #uncomment this to combine!!!!!!!!!!!!
    #plot_combined(layerruns, histories)
    #uncomment to save png!!!!!!!!
    #save_results_table(results)

    #best model
    best_idx  = max(range(len(results)),key=lambda i: results[i]["FreeMatch-F1"])
    best_r    = results[best_idx]
    best_ckpt = checkpoints[best_idx]

    #final summary as given in assi
    summary_lines = [
        "=" * 62,
        "  RESULTS SUMMARY — BiLSTM + Bahdanau Attention (GloVe 200d)",
        "=" * 62,
        f"  {'Layers (L)':<12} {'FreeMatch-F1':>14} {'Strict EM':>12} {'Precision':>11} {'Recall':>9}",
        "  " + "-" * 60,
    ]
    for r in results:
        marker = " ★" if r["L"] == best_r["L"] else "  "
        summary_lines.append(
            f"  L={r['L']}{marker}        "
            f"{r['FreeMatch-F1']:>14.4f}"
            f"{r['Strict-EM']:>12.4f}"
            f"{r['Precision']:>11.4f}"
            f"{r['Recall']:>9.4f}"
        )

    #for demo
    print("\nInference demo (best model)")
    best_model, _ = load_model_from_ckpt(bestmodelpath)
    demo_tokens = ["Accident", "Azadpur", "Delhi", "Mai", "Hua", "Hai"]
    pred_labels = predict(best_model, vocab, demo_tokens)
    print(f"  Tokens : {demo_tokens}")
    print(f"  Labels : {pred_labels}")

if __name__ == "__main__":
    main()