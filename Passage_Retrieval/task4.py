"""
Task 4: Query Likelihood Language Models
Here we implement Laplace, Lidstone (ε=0.1), and Dirichlet (μ=50) smoothing.
Outputs laplace.csv, lidstone.csv, dirichlet.csv with log-probability scores.
All scores are natural logarithm of P(Q|D).
"""
import re
import math
import pickle
import collections
import csv

TOKEN_RE = re.compile(r'[a-z]+')

def tokenize(text: str) -> list:
    return TOKEN_RE.findall(text.lower())

# hyperparameters
EPSILON = 0.1   # Lidstone correction
MU      = 50.0  # Dirichlet prior

# defining utility functions
def load_index(path: str) -> dict:
    with open(path, 'rb') as fh:
        return pickle.load(fh)

def load_vocab_size(path: str) -> int:
    """Return |V|: number of terms in vocabulary.txt (produced by task1)."""
    count = 0
    with open(path, 'r', encoding='utf-8') as fh:
        for line in fh:
            if line.strip():
                count += 1
    return count

def load_queries(path: str) -> list:
    queries = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as fh:
        for line in fh:
            parts = line.rstrip('\n').split('\t')
            if len(parts) >= 2:
                queries.append((int(parts[0]), parts[1]))
    return queries

def load_candidates(path: str) -> dict:
    """Return {qid: [pid, …]}, unique pids in encounter order."""
    cands = collections.defaultdict(list)
    seen  = collections.defaultdict(set)
    with open(path, 'r', encoding='utf-8', errors='ignore') as fh:
        for line in fh:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 2:
                continue
            qid = int(parts[0])
            pid = int(parts[1])
            if pid not in seen[qid]:
                seen[qid].add(pid)
                cands[qid].append(pid)
    return dict(cands)

def write_results(rows: list, path: str):
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh, lineterminator='\n')
        for qid, pid, score in rows:
            writer.writerow([qid, pid, score])

# Smoothed language model scorers 
def laplace_log_score(query_toks: list, pid: int, index: dict,
                      doc_len: int, vocab_size: int) -> float:
    """P_Lap(w|D) = (tf(w,D)+1) / (|D|+|V|); return Σ log P_Lap."""
    denom = doc_len + vocab_size
    score = 0.0
    for term in query_toks:
        tf     = index.get(term, {}).get(pid, 0)
        score += math.log((tf + 1) / denom)
    return score

def lidstone_log_score(query_toks: list, pid: int, index: dict,
                       doc_len: int, vocab_size: int,
                       eps: float = EPSILON) -> float:
    """P_Lid(w|D) = (tf(w,D)+ε) / (|D|+ε|V|); return Σ log P_Lid."""
    denom = doc_len + eps * vocab_size
    score = 0.0
    for term in query_toks:
        tf     = index.get(term, {}).get(pid, 0)
        score += math.log((tf + eps) / denom)
    return score

def dirichlet_log_score(query_toks: list, pid: int, index: dict,
                        doc_len: int, coll_probs: dict,
                        mu: float = MU) -> float:
    """P_Dir(w|D) = (tf(w,D)+μP(w|C)) / (|D|+μ); return Σ log P_Dir."""
    denom = doc_len + mu
    score = 0.0
    for term in query_toks:
        tf     = index.get(term, {}).get(pid, 0)
        p_coll = coll_probs.get(term, 1e-15)
        prob   = (tf + mu * p_coll) / denom
        score += math.log(max(prob, 1e-300))
    return score

# Main 
def main():
    for prereq in ['inverted_index.pkl','vocabulary.txt','test-queries.tsv','candidate-passages-top1000.tsv']:
        if not __import__('os').path.exists(prereq):
            raise FileNotFoundError(f"'{prereq}' not found. Run task1.py then task2.py first.")
    idx          = load_index('inverted_index.pkl')
    vocab_size   = load_vocab_size('vocabulary.txt')
    queries      = load_queries('test-queries.tsv')
    cands_map    = load_candidates('candidate-passages-top1000.tsv')

    index        = idx['index']
    doc_lengths  = idx['doc_lengths']
    total_tokens = idx['total_tokens']

    # collection language model P(w|C) = cf(w) / total_tokens
    coll_probs = {t: cf / total_tokens
                  for t, cf in idx['collection_freq'].items()}

    print(f'Vocabulary size |V|  : {vocab_size:,}')
    print(f'Collection tokens    : {total_tokens:,}')

    lap_rows, lid_rows, dir_rows = [], [], []

    for qid, query_text in queries:
        candidate_pids = cands_map.get(qid, [])
        if not candidate_pids:
            continue
        query_toks = tokenize(query_text)

        lap_sc, lid_sc, dir_sc = {}, {}, {}
        for pid in candidate_pids:
            dl            = doc_lengths.get(pid, 0)
            lap_sc[pid]   = laplace_log_score(query_toks,  pid, index, dl, vocab_size)
            lid_sc[pid]   = lidstone_log_score(query_toks, pid, index, dl, vocab_size)
            dir_sc[pid]   = dirichlet_log_score(query_toks, pid, index, dl, coll_probs)

        def top100(sc: dict) -> list:
            return sorted(candidate_pids, key=lambda p: sc[p], reverse=True)[:100]

        for pid in top100(lap_sc):
            lap_rows.append((qid, pid, lap_sc[pid]))
        for pid in top100(lid_sc):
            lid_rows.append((qid, pid, lid_sc[pid]))
        for pid in top100(dir_sc):
            dir_rows.append((qid, pid, dir_sc[pid]))

    write_results(lap_rows, 'laplace.csv')
    write_results(lid_rows, 'lidstone.csv')
    write_results(dir_rows, 'dirichlet.csv')

    print(f'laplace.csv   rows : {len(lap_rows):,}')
    print(f'lidstone.csv  rows : {len(lid_rows):,}')
    print(f'dirichlet.csv rows : {len(dir_rows):,}')

if __name__ == '__main__':
    main()
