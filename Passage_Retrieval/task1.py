"""
Task 1: Text Statistics
This code analyses term distributions in passage-collection.txt and compares with Zipf's law.
Outputs: vocabulary.txt, figure1.pdf, figure2.pdf, figure3.pdf
"""
import re
import collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TOKEN_RE = re.compile(r'[a-z]+')

def tokenize(text: str) -> list:
    """Lowercase and extract contiguous alphabetic tokens."""
    return TOKEN_RE.findall(text.lower())

# Stop words: nltk.corpus.stopwords.words('english'), 127 terms (used for stop-word-removal comparison; otherwise retained in this task)
STOP_WORDS = frozenset([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
    'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
    'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
    'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who',
    'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
    'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the',
    'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
    'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
    'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to',
    'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
    'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
    'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
    'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
    'only', 'own', 'same', 'so', 'than', 'too', 'very', 's',
    't', 'can', 'will', 'just', 'don', 'should', 'now'
])

def zipf_probs(N: int, ranks: np.ndarray, s: float = 1.0) -> np.ndarray:
    """Zipf probabilities f(k;s,N) = k^{-s} / sum_{i=1}^{N} i^{-s}."""
    raw = ranks ** (-s)
    H_N = float(np.sum(np.arange(1, N + 1, dtype=np.float64) ** (-s)))
    return raw / H_N

def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))

def estimate_s(ranks: np.ndarray, freqs: np.ndarray) -> float:
    """OLS slope in log-log space."""
    lo = max(1, int(len(ranks) * 0.01))
    hi = int(len(ranks) * 0.99)
    slope, _ = np.polyfit(np.log(ranks[lo:hi]), np.log(freqs[lo:hi]), 1)
    return -slope

def plot_setup(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=8)

def main():
    # term counts from full passage collection
    term_counts = collections.Counter()
    with open('passage-collection.txt', 'r', encoding='utf-8', errors='ignore') as fh:
        for line in fh:
            term_counts.update(tokenize(line))

    N = len(term_counts)
    print(f'Vocabulary size (stop words included) : {N:,}')

    with open('vocabulary.txt', 'w', encoding='utf-8') as fh:
        for term in sorted(term_counts):
            fh.write(term + '\n')

    # sorted empirical distributions
    sorted_terms  = term_counts.most_common()
    total_tokens  = sum(term_counts.values())
    ranks         = np.arange(1, N + 1, dtype=np.float64)
    emp_f         = np.array([c / total_tokens for _, c in sorted_terms])
    zipf_s1       = zipf_probs(N, ranks, s=1.0)

    # quantify deviation
    mse_sw  = mse(emp_f, zipf_s1)
    s_hat   = estimate_s(ranks, emp_f)
    zipf_sh = zipf_probs(N, ranks, s=s_hat)

    print(f'Total tokens                          : {total_tokens:,}')
    print(f'MSE  (empirical vs Zipf s=1)          : {mse_sw:.4e}')
    print(f'Estimated Zipf exponent               : {s_hat:.4f}')
    for label, lo, hi in [('Head',   0,              int(N * 0.01)),
                           ('Body',    int(N * 0.01),  int(N * 0.50)),
                           ('Tail',  int(N * 0.50),  N)]:
        print(f'  MSE {label}: {mse(emp_f[lo:hi], zipf_s1[lo:hi]):.4e}')

    # plot Fig 1 (linear scale)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(ranks, emp_f, lw=0.6, label='Empirical')
    plot_setup(ax, 'Frequency rank', 'Probability of occurrence',
               'Term frequency distribution')
    plt.tight_layout()
    fig.savefig('figure1.pdf')
    plt.close(fig)

    # plot Fig 2: log-log with Zipf s=1
    
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.loglog(ranks, emp_f,    lw=0.8, label='Empirical')
    ax.loglog(ranks, zipf_s1,  lw=0.8, ls='--', label='Zipf ($s=1$)')
    plot_setup(ax, 'Rank (log)', 'Probability (log)',
               "Empirical vs Zipf's law")
    plt.tight_layout()
    fig.savefig('figure2.pdf')
    plt.close(fig)

    # stop-word-removed distribution
    tc_nsw       = collections.Counter({t: c for t, c in term_counts.items()
                                         if t not in STOP_WORDS})
    sorted_nsw   = tc_nsw.most_common()
    N_nsw        = len(sorted_nsw)
    total_nsw    = sum(tc_nsw.values())
    ranks_nsw    = np.arange(1, N_nsw + 1, dtype=np.float64)
    emp_nsw      = np.array([c / total_nsw for _, c in sorted_nsw])
    zipf_nsw     = zipf_probs(N_nsw, ranks_nsw, s=1.0)
    s_nsw        = estimate_s(ranks_nsw, emp_nsw)

    mse_nsw = mse(emp_nsw, zipf_nsw)
    print(f'\nVocabulary size (stop words removed)  : {N_nsw:,}')
    print(f'MSE  (no stop words, vs Zipf s=1)     : {mse_nsw:.4e}')
    print(f'Estimated Zipf exponent (no SW)       : {s_nsw:.4f}')
    print(f'MSE reduction from removing stop words: '
          f'{(mse_sw - mse_nsw) / mse_sw * 100:.1f}%')

    # fig 3: log-log without stop words 
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.loglog(ranks_nsw, emp_nsw,  lw=0.8, label='Empirical (no stop words)')
    ax.loglog(ranks_nsw, zipf_nsw, lw=0.8, ls='--', label='Zipf ($s=1$)')
    plot_setup(ax, 'Rank (log)', 'Probability (log)',
               "Distribution after stop word removal")
    plt.tight_layout()
    fig.savefig('figure3.pdf')
    plt.close(fig)

    print(f'\nTop-5 terms with stop-words included: {[t for t, _ in sorted_terms[:5]]}')
    print(f'Top-5 terms without stop-words: {[t for t, _ in sorted_nsw[:5]]}')

if __name__ == '__main__':
    main()
