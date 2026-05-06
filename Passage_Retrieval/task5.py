"""
COMP0084 Coursework – Task 5: Retrieval Evaluation & Machine Learning

D13 – Own AP and NDCG. BM25 grid-search on train-data.tsv; eval on validation-data.tsv.
D14 – FastText embeddings (averaged); logistic regression self-implementde; LR analysis;
      evaluation on validation-data.tsv.
D15 – All results printed 

To avoid RAM crashes on a 12GB RAM machine, we incorporate the following:
- Passage embeddings cached as float16 in RAM, as is validation feature matrix
- Training feature matrix written to a memory-mapped file on disk rather than held in RAM
- predict_proba uses batched inference, casting to float64 per bath
- Finally we use staged del + gc.collect () to free training data, training pid cache, and FastText model as soon as they are no longer needed, to avoid crashes at validation stage

To reduce the run time to below one hour:
- Each unique passage is embedded exactly once regardless of how many queries it appears against
- Query embeddings use numpy fancy indexing into model.vectors t avoid per-token Python array copies
- Validation BM25 evaluates both default and optimised parameters in a single pass (rather than two separate iterations)
- Training memmap read sequentially during SGD (no random access between epochs)
"""
import re, sys, math, pickle, collections, os, gc
import numpy as np

TOKEN_RE = re.compile(r'[a-z]+')
FASTTEXT_LOCAL_PATH = None
MMAP_FILE = 'X_tr_tmp.dat'

def tokenize(text): return TOKEN_RE.findall(text.lower())

# defining evaluation metrics (D13)

def average_precision(ranked: list, relevant: set) -> float:
    """AP = (1/|R|) Σ P@k for k where d_k ∈ R. Returns 0 if |R|=0."""
    R = len(relevant)
    if R == 0: return 0.0
    hits = ap = 0
    for k, pid in enumerate(ranked, 1):
        if pid in relevant:
            hits += 1; ap += hits / k
    return ap / R

def dcg_at_k(ranked: list, rel: dict, k: int) -> float:
    """DCG@k = Σ rel(d_i)/log2(i+1). Binary: equals (2^rel-1)/log2(i+1)."""
    return sum(rel.get(pid, 0) / math.log2(i + 1)
               for i, pid in enumerate(ranked[:k], 1))

def ndcg_at_k(ranked: list, rel: dict, k: int = 10) -> float:
    """NDCG@k = DCG@k / IDCG@k. Returns 0 if IDCG=0."""
    ideal = sorted(rel.values(), reverse=True)
    idcg  = sum(v / math.log2(i + 1) for i, v in enumerate(ideal[:k], 1))
    return dcg_at_k(ranked, rel, k) / idcg if idcg > 0 else 0.0

# 2. BM25

def bm25_rank(qtoks: list, cands: list, idx: dict,
              k1: float, b: float, k2: float = 100.0) -> list:
    index, df_map, dl_map = idx['index'], idx['doc_freq'], idx['doc_lengths']
    N, avgdl = idx['N'], idx['avgdl']
    qtf = collections.Counter(qtoks)
    scores = {p: 0.0 for p in cands}; cset = set(cands)
    for term, qf in qtf.items():
        if term not in index: continue
        df  = df_map[term]; idf = math.log((N - df + 0.5) / (df + 0.5))
        qfw = (qf * (k2 + 1)) / (qf + k2)
        for pid, tf in index[term].items():
            if pid not in cset: continue
            K = k1 * ((1 - b) + b * dl_map[pid] / avgdl)
            scores[pid] += idf * ((tf * (k1 + 1)) / (tf + K)) * qfw
    return sorted(cands, key=lambda p: scores[p], reverse=True)

def eval_bm25(data: dict, idx: dict, k1: float, b: float) -> tuple:
    aps, ndcgs = [], []
    for qd in data.values():
        cands  = list(qd['passages']); qtoks = tokenize(qd['query'])
        rel    = {pid: r for pid, (_, r) in qd['passages'].items()}
        ranked = bm25_rank(qtoks, cands, idx, k1, b)
        aps.append(average_precision(ranked, {p for p, r in rel.items() if r > 0}))
        ndcgs.append(ndcg_at_k(ranked, rel))
    return float(np.mean(aps)), float(np.mean(ndcgs))

def eval_bm25_two(data: dict, idx: dict,
                  k1a: float, ba: float, k1b: float, bb: float) -> tuple:
    """Evaluate two BM25 configs in one pass — avoids double val iteration."""
    aps_a, ndcgs_a, aps_b, ndcgs_b = [], [], [], []
    for qd in data.values():
        cands = list(qd['passages']); qtoks = tokenize(qd['query'])
        rel   = {pid: r for pid, (_, r) in qd['passages'].items()}
        rset  = {p for p, r in rel.items() if r > 0}
        for aps, ndcgs, k1, b in [(aps_a, ndcgs_a, k1a, ba),
                                   (aps_b, ndcgs_b, k1b, bb)]:
            ranked = bm25_rank(qtoks, cands, idx, k1, b)
            aps.append(average_precision(ranked, rset))
            ndcgs.append(ndcg_at_k(ranked, rel))
    return (float(np.mean(aps_a)), float(np.mean(ndcgs_a)),
            float(np.mean(aps_b)), float(np.mean(ndcgs_b)))

def optimise_bm25(train: dict, idx: dict) -> tuple:
    K1_GRID = [0.5, 1.0, 1.2, 1.5, 2.0]
    B_GRID  = [0.0, 0.25, 0.50, 0.75, 1.0]
    best_map, best_k1, best_b = -1.0, 1.2, 0.75
    print(f'  {"k1":>5}  {"b":>5}  {"MAP":>8}')
    for k1 in K1_GRID:
        for b in B_GRID:
            m, _ = eval_bm25(train, idx, k1, b)
            print(f'  {k1:>5.2f}  {b:>5.2f}  {m:>8.4f}')
            if m > best_map: best_map, best_k1, best_b = m, k1, b
    print(f'  Best: k1={best_k1}  b={best_b}  MAP={best_map:.4f}')
    return best_k1, best_b

#loading data

def load_index(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"'{path}' not found — run task2.py first.")
    with open(path, 'rb') as f: return pickle.load(f)

def load_labelled(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"'{path}' not found.")
    data = {}
    with open(path, 'r', encoding='utf-8', errors='ignore') as fh:
        next(fh)
        for line in fh:
            p = line.rstrip('\n').split('\t')
            if len(p) < 5: continue
            qid, pid = int(p[0]), int(p[1])
            if qid not in data: data[qid] = {'query': p[2], 'passages': {}}
            data[qid]['passages'][pid] = (p[3], int(float(p[4])))
    return data

# word embeddings (D14)

def load_embeddings(local_path=None):
    if len(sys.argv) > 1: local_path = sys.argv[1]
    if local_path and os.path.exists(local_path):
        import gensim.models
        print(f'Loading local FastText: {local_path}')
        return gensim.models.fasttext.load_facebook_model(local_path).wv
    import gensim.downloader as api
    print('Downloading FastText (~1 GB, cached after first run) ...')
    return api.load('fasttext-wiki-news-subwords-300')

def avg_embed(tokens: list, model, dim: int = 300) -> np.ndarray:
    """
    Vectorised mean via numpy fancy-index into model.vectors.
    Single C-level slice — avoids per-token Python array copies.
    """
    if not tokens: return np.zeros(dim, dtype=np.float32)
    idxs = [model.key_to_index[t] for t in tokens if t in model.key_to_index]
    if not idxs: return np.zeros(dim, dtype=np.float32)
    return model.vectors[idxs].mean(axis=0).astype(np.float32)

def build_pid_cache_f16(data: dict, model, dim: int, label: str) -> dict:
    """
    Cache passage embeddings by pid as FLOAT16.
    float16 is essential: float32 training cache = 3.52 GB → OOM.
                          float16 training cache = 1.76 GB → safe.
    Each unique passage embedded exactly once (dedup saves ~33% time).
    """
    pid_to_text = {}
    for d in data.values():
        for pid, (text, _) in d['passages'].items():
            if pid not in pid_to_text: pid_to_text[pid] = text
    n_unique = len(pid_to_text)
    n_total  = sum(len(d['passages']) for d in data.values())
    print(f'  {n_unique:,} unique pids / {n_total:,} total '
          f'(rep {n_total / max(n_unique, 1):.1f}x)')
    cache = {}
    for i, (pid, text) in enumerate(pid_to_text.items()):
        cache[pid] = avg_embed(tokenize(text), model, dim).astype(np.float16)
        if i % 100_000 == 0 and i > 0:
            print(f'  {label} cache: {i:,}/{n_unique:,}')
    return cache

# add cosine similarity as interaction feature - feature width is now 2*dim+1
def _cosine_f16(q: np.ndarray, p: np.ndarray) -> np.float16:
    """Cosine similarity between two float16 vectors; computed in float32."""
    q32 = q.astype(np.float32); p32 = p.astype(np.float32)
    denom = np.linalg.norm(q32) * np.linalg.norm(p32)
    return np.float16(float(np.dot(q32, p32) / (denom + 1e-10)))

def write_train_memmap(data: dict, q_cache: dict, p_cache: dict,
                       dim: int, path: str) -> tuple:
    """Write float16 training feature matrix via O(1) pid cache lookup."""
    n = sum(len(d['passages']) for d in data.values())
    X = np.memmap(path, dtype='float16', mode='w+', shape=(n, 2 * dim + 1))  # CHANGED: +1
    y = np.zeros(n, dtype='float32')
    i = 0
    for j, (qid, d) in enumerate(data.items()):
        q = q_cache[qid]
        for pid, (_, rel) in d['passages'].items():
            p = p_cache[pid]
            X[i, :dim] = q
            X[i, dim:2*dim] = p                   # explicit slice
            X[i, 2*dim] = _cosine_f16(q, p)       # cosine interaction
            y[i] = float(rel); i += 1
        if j % 1000 == 0:
            print(f'  Written {j}/{len(data)} queries ({i:,} samples) ...')
    del X
    return y, n

def build_val_features(data: dict, q_cache: dict, p_cache: dict,
                       dim: int) -> tuple:
    """Float16 val matrix in RAM — slightly over 1.32 GB for 601 dims."""
    n = sum(len(d['passages']) for d in data.values())
    X = np.zeros((n, 2 * dim + 1), dtype='float16')  # CHANGED: +1
    y = np.zeros(n, dtype='float32'); meta = []; i = 0
    zero_p = np.zeros(dim, dtype='float16')
    for qid, d in data.items():
        q = q_cache[qid]
        for pid, (_, rel) in d['passages'].items():
            p = p_cache.get(pid, zero_p)
            X[i, :dim] = q
            X[i, dim:2*dim] = p                   # explicit slice
            X[i, 2*dim] = _cosine_f16(q, p)       # cosine interaction
            y[i] = float(rel); meta.append((qid, pid)); i += 1
    return X, y, meta


# logistic regression implementation (D14)

def sigmoid(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos]); out[~pos] = ez / (1.0 + ez)
    return out

class LogisticRegression:
    """
    Mini-batch SGD logistic regression, weighted binary cross-entropy loss.
    Batch size 512. No epoch shuffling (sequential memmap reads).
    Accepts float16 input; casts to float64 per-batch for stability.
    predict_proba uses BATCHED inference to avoid high-RAM float64
    temporary that would otherwise crash RAM.
    pos_weight upweights positive examples to address class imbalance.  
    """
    def __init__(self, lr=0.01, epochs=5, batch_size=512, seed=0,
                 pos_weight=1.0):                                        # pos_weight
        self.lr, self.epochs, self.batch = lr, epochs, batch_size
        self.seed = seed; self.w = None; self.b = 0.0
        self.pos_weight = pos_weight                            

    def fit(self, X, y) -> float:
        """Train; return training weighted BCE on final epoch."""
        n, d = X.shape
        self.w = np.zeros(d, dtype=np.float64); self.b = 0.0
        y64 = y.astype(np.float64); is_mmap = isinstance(X, np.memmap)
        rng = np.random.default_rng(self.seed)
        last_p, last_y = [], []
        for epoch in range(self.epochs):
            last = (epoch == self.epochs - 1)
            perm = None if is_mmap else rng.permutation(n)
            for s in range(0, n, self.batch):
                idx = (np.arange(s, min(s + self.batch, n)) if is_mmap
                       else perm[s:s + self.batch])
                Xb = X[idx].astype(np.float64); yb = y64[idx]
                pr = sigmoid(Xb @ self.w + self.b)
                # addressing imbalance: weight positive examples by pos_weight
                w_vec = np.where(yb == 1, self.pos_weight, 1.0)
                err = w_vec * (pr - yb)
                self.w -= self.lr * (Xb.T @ err) / len(idx)
                self.b -= self.lr * err.mean()
                if last: last_p.extend(pr.tolist()); last_y.extend(yb.tolist())
        EPS = 1e-15
        p = np.clip(np.array(last_p), EPS, 1 - EPS); lbl = np.array(last_y)
        # weighted BCE for reporting
        w_arr = np.where(lbl == 1, self.pos_weight, 1.0)
        return -float(np.mean(w_arr * (lbl * np.log(p) + (1 - lbl) * np.log(1 - p))))

    def predict_proba(self, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        """
        Batched inference: processes X in chunks of 4096 rows.
        Required to prevent crash at evaluation.
        """
        n = X.shape[0]
        out = np.empty(n, dtype=np.float64)
        for s in range(0, n, batch_size):
            e = min(s + batch_size, n)
            out[s:e] = sigmoid(X[s:e].astype(np.float64) @ self.w + self.b)
        return out

def eval_logreg(data: dict, meta: list, proba: np.ndarray) -> tuple:
    groups = collections.defaultdict(list)
    for (qid, pid), prob in zip(meta, proba):
        groups[qid].append((pid, float(prob)))
    aps, ndcgs = [], []
    for qid, pid_scores in groups.items():
        rel    = {pid: r for pid, (_, r) in data[qid]['passages'].items()}
        ranked = [p for p, _ in sorted(pid_scores, key=lambda x: -x[1])]
        aps.append(average_precision(ranked, {p for p, r in rel.items() if r > 0}))
        ndcgs.append(ndcg_at_k(ranked, rel))
    return float(np.mean(aps)), float(np.mean(ndcgs))

#main

def main():
    idx   = load_index('inverted_index.pkl')
    train = load_labelled('train-data.tsv')
    val   = load_labelled('validation-data.tsv')

    # D13: BM25
    print('BM25 grid search on train-data.tsv:')
    best_k1, best_b = optimise_bm25(train, idx)

    base_map, base_ndcg, opt_map, opt_ndcg = eval_bm25_two(
        val, idx, 1.2, 0.75, best_k1, best_b)
    print('\nValidation performance:')
    print(f'  BM25 default  (k1=1.2, b=0.75)      '
          f'MAP={base_map:.4f}  NDCG@10={base_ndcg:.4f}')
    print(f'  BM25 optimised (k1={best_k1}, b={best_b})  '
          f'MAP={opt_map:.4f}  NDCG@10={opt_ndcg:.4f}')

    # D14: Embeddings + Logistic Regression
    print('Word embeddings + logistic regression:')

    DIM   = 300
    model = load_embeddings(FASTTEXT_LOCAL_PATH)

    print('Pre-computing query embeddings ...')
    q_cache_tr = {qid: avg_embed(tokenize(d['query']), model, DIM)
                  for qid, d in train.items()}
    q_cache_va = {qid: avg_embed(tokenize(d['query']), model, DIM)
                  for qid, d in val.items()}

    # Training pid cache in float16
    print('Pre-computing training passage embeddings (float16 pid cache) ...')
    p_cache_tr = build_pid_cache_f16(train, model, DIM, 'Train')

    print('Writing float16 training feature matrix to disk ...')
    y_tr, n_tr = write_train_memmap(train, q_cache_tr, p_cache_tr, DIM, MMAP_FILE)
    print(f'  Training samples written: {n_tr:,}')

    # free train data and cache
    del train, q_cache_tr, p_cache_tr
    gc.collect()

    # val pid cache in float16
    print('Pre-computing validation passage embeddings (float16 pid cache) ...')
    p_cache_va = build_pid_cache_f16(val, model, DIM, 'Val')

    del model; gc.collect()

    print('Building float16 validation feature matrix ...')
    X_va, y_va, meta_va = build_val_features(val, q_cache_va, p_cache_va, DIM)
    print(f'  Validation samples: {X_va.shape[0]:,}')
    del p_cache_va, q_cache_va; gc.collect()

    X_tr = np.memmap(MMAP_FILE, dtype='float16', mode='r',
                     shape=(n_tr, 2 * DIM + 1))                        # CHANGED: +1

    # LR sweep: 2 epochs each (sufficient to rank LRs; 5 models × 2 = 10 epochs)
    LRS = [1e-4, 1e-3, 1e-2, 0.1, 0.3]
    print(f'\n  {"LR":<8} {"Train BCE":>10} {"Val MAP":>9} {"Val NDCG@10":>12}')
    best_lr, best_val_map = 0.01, -1.0
    for lr in LRS:
        clf = LogisticRegression(lr=lr, epochs=2, batch_size=512,
                                 pos_weight=10.0)                      # added pos_weight
        bce = clf.fit(X_tr, y_tr)
        # predict_proba is batched
        vm, vn = eval_logreg(val, meta_va, clf.predict_proba(X_va))
        print(f'  {lr:<8.0e} {bce:>10.4f} {vm:>9.4f} {vn:>12.4f}')
        if vm > best_val_map: best_val_map, best_lr = vm, lr

    print(f'\n  Best learning rate: {best_lr}')
    print('Training final model (best LR, 3 epochs) ...')
    final = LogisticRegression(lr=best_lr, epochs=3, batch_size=512,
                                pos_weight=10.0)                
    final.fit(X_tr, y_tr)
    fm, fn = eval_logreg(val, meta_va, final.predict_proba(X_va))

    print(f'\nFinal results on validation-data.tsv:')
    print(f'  LogReg (lr={best_lr}, 3 epochs)  MAP={fm:.4f}  NDCG@10={fn:.4f}')

    print('\nSummary')
    print(f'  BM25 default                    MAP={base_map:.4f}  NDCG@10={base_ndcg:.4f}')
    print(f'  BM25 optimised (k1={best_k1}, b={best_b})  '
          f'MAP={opt_map:.4f}  NDCG@10={opt_ndcg:.4f}')
    print(f'  LogReg (lr={best_lr}, 3 epochs)  MAP={fm:.4f}  NDCG@10={fn:.4f}')

    del X_tr
    if os.path.exists(MMAP_FILE): os.remove(MMAP_FILE)

if __name__ == '__main__':
    main()
