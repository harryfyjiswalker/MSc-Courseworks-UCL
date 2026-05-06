import os
"""
Task 2: Inverted Index
Builds an inverted index over unique passages in candidate-passages-top1000.tsv.
Loads vocabulary from task1 output (vocabulary.txt).
Outputs: inverted_index.pkl
"""
import re
import collections
import pickle

TOKEN_RE = re.compile(r'[a-z]+')

def tokenize(text: str) -> list:
    return TOKEN_RE.findall(text.lower())

def build_index(passage_file: str, vocab: set) -> dict:
    """
    Returns a dict with:
      index           : {term: {pid: raw_tf}}
      doc_freq        : {term: df}
      doc_lengths     : {pid: token_count}
      collection_freq : {term: total_count_across_all_passages}
      total_tokens    : int
      N               : int  (number of unique passages)
      avgdl           : float
    """
    # collect unique passages (here we make first occurence win for duplicates)
    unique_passages = {}
    with open(passage_file, 'r', encoding='utf-8', errors='ignore') as fh:
        for line in fh:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 4:
                continue
            pid = int(parts[1])
            if pid not in unique_passages:
                unique_passages[pid] = parts[3]

    index           = collections.defaultdict(dict)
    doc_lengths     = {}
    collection_freq = collections.Counter()

    for pid, text in unique_passages.items():
        tokens = tokenize(text)
        doc_lengths[pid] = len(tokens)
        for term, tf in collections.Counter(tokens).items():
            if term in vocab:
                index[term][pid]      = tf
                collection_freq[term] += tf

    index    = dict(index)
    doc_freq = {t: len(p) for t, p in index.items()}
    N        = len(unique_passages)
    tot      = sum(doc_lengths.values())
    avgdl    = tot / N if N else 1.0

    return {
        'index'          : index,
        'doc_freq'       : doc_freq,
        'doc_lengths'    : doc_lengths,
        'collection_freq': dict(collection_freq),
        'total_tokens'   : tot,
        'N'              : N,
        'avgdl'          : avgdl,
    }

def main():
    if not os.path.exists('vocabulary.txt'):
        raise FileNotFoundError(
            "'vocabulary.txt' not found. Run task1.py first.")
    vocab = set()
    with open('vocabulary.txt', 'r', encoding='utf-8') as fh:
        for line in fh:
            t = line.strip()
            if t:
                vocab.add(t)
    print(f'Vocabulary loaded   : {len(vocab):,} terms')

    data = build_index('candidate-passages-top1000.tsv', vocab)

    print(f'Unique passages     : {data["N"]:,}')
    print(f'Index terms         : {len(data["index"]):,}')
    print(f'Total tokens        : {data["total_tokens"]:,}')
    print(f'Average document length  : {data["avgdl"]:.2f}')

    with open('inverted_index.pkl', 'wb') as fh:
        pickle.dump(data, fh, protocol=4)
    print('inverted_index.pkl saved.')

if __name__ == '__main__':
    main()
