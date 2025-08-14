import sys
import argparse
from NR_CINOBERTScore import *


def cmd_line_input_tib2zh():
    scorer = TibZhBERTScorer()

    # Change this
    mt, src, _ = sys.argv[1:]
    chinese_translations, tibetan_sources = [], []
    
    with open(mt, 'r', encoding='utf-8') as f1, \
         open(src, 'r', encoding='utf-8') as f2:
        for l1, l2 in zip(f1, f2):
            chinese_translations.append(l1.strip())
            tibetan_sources.append(l2.strip())

    results_tib2zh = scorer.evaluate_ref_free_dataset(
        tibetan_sources,
        chinese_translations,
        direction="tib2zh",
        batch_size=2
    )
    
    return results_tib2zh


def main():
    """
    Usage: python3 TiBERTScore/ref_free_BERTScorer.py <hyp> <src> <outpath>
    python CINOBERTScore/NR_CINOBERTScorer.py CINOBERTScore/tbt-cn-200/mt-hyps/hyp_deepseek-v3 TiBERTScore/tbt-cn-200/src_clean.txt CINOBERTScore/tbt-cn-200/NR_BERTScore_mev_scores/NR_BERTScore_deepseek-v3
    """

    # change this
    results = cmd_line_input_tib2zh()
    # results = example_usage_tib2zh() 

    outfile = sys.argv[3]

    # Print overall avg
    print(f"Average F1 Score: {results['f1']['mean']:.4f} ± {results['f1']['std']:.4f}")
    print(f"Average Precision: {results['precision']['mean']:.4f}")
    print(f"Average Recall: {results['recall']['mean']:.4f}")

    # Print F1 line by line
    scores = results['f1']['scores']
    with open(outfile, 'w+') as f:
        for score in scores:
            f.write(f"{(score * 4 + 1):.4f}\n")


if __name__ == "__main__":
    main()