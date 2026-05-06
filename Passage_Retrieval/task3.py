"""
Task 3: Retrieval Models
TF-IDF (cosine similarity) and BM25 re-ranking over candidate passages; the queries are in in test-queries.tsv order (top 100 per query).
Outputs: tfidf.csv and bm25.csv, with format qid,pid,score
"""
import re
import math
import pickle
import collections
import csv

TOKEN_RE = re.compile(r'[a-z]+')

def tokenize(text: str) -> list:
    return TOKEN_RE.findall(text.lower())

# BM25 hyperparameters (from CW)
K1 = 1.2
K2 = 100
B  = 0.75

def load_index(path: str) -> dict:
    with open(path, 'rb') as fh:
        return pickle.load(fh)

def load_queries(path: str) -> list:
    rows = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as fh:
        for line in fh:
            p = line.rstrip('\n').split('\t')
            if len(p) >= 2:
                rows.append((int(p[0]), p[1]))
    return rows

def load_candidates(path: str) -> dict:
    """Return {qid: [pid, …]} preserving first-encounter order, unique pids."""
    cands = collections.defaultdict(list)
    seen  = collections.defaultdict(set)
    with open(path, 'r', encoding='utf-8', errors='ignore') as fh:
        for line in fh:
            p = line.rstrip('\n').split('\t')
            if len(p) < 2:
                continue
            qid, pid = int(p[0]), int(p[1])
            if pid not in seen[qid]:
                seen[qid].add(pid)
                cands[qid].append(pid)
    return dict(cands)

def write_csv(rows: list, path: str):
    """Write list of (qid, pid, score) to CSV without header."""
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh, lineterminator='\n')
        for qid, pid, score in rows:
            w.writerow([qid, pid, score])

# building tf-idf
def build_doc_tfidf_norms(idx: dict) -> dict:
    """Precompute L2 norm of the TF-IDF vector for every passage"""
    index    = idx['index']
    df_map   = idx['doc_freq']
    dl_map   = idx['doc_lengths']
    N        = idx['N']
    sq_norms = collections.defaultdict(float)
    for term, postings in index.items():
        idf = math.log(N / df_map[term])
        for pid, tf in postings.items():
            w = (tf / dl_map[pid]) * idf
            sq_norms[pid] += w * w
    return {pid: math.sqrt(v) for pid, v in sq_norms.items()}

def score_tfidf(q_toks: list, cand_set: set, idx: dict,
                doc_norms: dict) -> dict:
    """Return {pid: cosine_sim} for passages in cand_set"""
    index  = idx['index']
    df_map = idx['doc_freq']
    dl_map = idx['doc_lengths']
    N      = idx['N']

    qtf  = collections.Counter(q_toks)
    qlen = len(q_toks)

    q_weights, q_norm_sq = {}, 0.0
    for term, tf_q in qtf.items():
        if term not in df_map:
            continue
        idf          = math.log(N / df_map[term])
        w            = (tf_q / qlen) * idf
        q_weights[term] = w
        q_norm_sq   += w * w

    if q_norm_sq == 0.0:
        return {pid: 0.0 for pid in cand_set}

    q_norm = math.sqrt(q_norm_sq)
    dot    = collections.defaultdict(float)
    for term, w_q in q_weights.items():
        if term not in index:
            continue
        idf = math.log(N / df_map[term])
        for pid, tf_d in index[term].items():
            if pid not in cand_set:
                continue
            dot[pid] += w_q * (tf_d / dl_map[pid]) * idf

    scores = {}
    for pid in cand_set:
        d  = dot.get(pid, 0.0)
        dn = doc_norms.get(pid, 0.0)
        scores[pid] = (d / (q_norm * dn)) if dn > 0.0 else 0.0
    return scores

# BM25 
def score_bm25(q_toks: list, cand_set: set, idx: dict) -> dict:
    """Return {pid: BM25_score} for passages in cand_set"""
    index  = idx['index']
    df_map = idx['doc_freq']
    dl_map = idx['doc_lengths']
    N      = idx['N']
    avgdl  = idx['avgdl']

    qtf    = collections.Counter(q_toks)
    scores = collections.defaultdict(float)
    for term, qf in qtf.items():
        if term not in index:
            continue
        df  = df_map[term]
        idf = math.log((N - df + 0.5) / (df + 0.5))
        qf_w = (qf * (K2 + 1)) / (qf + K2)
        for pid, tf in index[term].items():
            if pid not in cand_set:
                continue
            K     = K1 * ((1 - B) + B * dl_map[pid] / avgdl)
            tf_w  = (tf * (K1 + 1)) / (tf + K)
            scores[pid] += idf * tf_w * qf_w

    return {pid: scores.get(pid, 0.0) for pid in cand_set}

def main():
    for prereq in ['inverted_index.pkl','test-queries.tsv','candidate-passages-top1000.tsv']:
        if not __import__('os').path.exists(prereq):
            raise FileNotFoundError(f"'{prereq}' not found. Run task1.py then task2.py first.")
    idx       = load_index('inverted_index.pkl')
    queries   = load_queries('test-queries.tsv')
    cands_map = load_candidates('candidate-passages-top1000.tsv')

    doc_norms  = build_doc_tfidf_norms(idx)
    tfidf_rows = []
    bm25_rows  = []

    for qid, query_text in queries:
        cand_pids = cands_map.get(qid, [])
        if not cand_pids:
            continue
        cand_set  = set(cand_pids)
        q_toks    = tokenize(query_text)

        tf_sc = score_tfidf(q_toks, cand_set, idx, doc_norms)
        bm_sc = score_bm25(q_toks,  cand_set, idx)

        top100_tf = sorted(cand_pids, key=lambda p: tf_sc[p], reverse=True)[:100]
        top100_bm = sorted(cand_pids, key=lambda p: bm_sc[p], reverse=True)[:100]

        for pid in top100_tf:
            tfidf_rows.append((qid, pid, tf_sc[pid]))
        for pid in top100_bm:
            bm25_rows.append((qid, pid, bm_sc[pid]))

    write_csv(tfidf_rows, 'tfidf.csv')
    write_csv(bm25_rows,  'bm25.csv')
    print(f'tfidf.csv rows : {len(tfidf_rows):,}')
    print(f'bm25.csv  rows : {len(bm25_rows):,}')

if __name__ == '__main__':
    main()
