import csv
import os
from sklearn.metrics import f1_score

def f1_for_sample(gold_set, pred_set):
    """
    Compute F1 score for one sample given two sets.
    """
    if not gold_set and not pred_set:
        return 1.0
    common = gold_set.intersection(pred_set)
    precision = len(common) / len(pred_set) if pred_set else 0.0
    recall = len(common) / len(gold_set) if gold_set else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def read_txt_labels(file_path):
    """
    Reads a text file with each line formatted as:
      article_id<TAB>narrative_labels<TAB>subnarrative_labels
    Returns a dictionary mapping article_id to a tuple:
      (set of (narrative, subnarrative) pairs, set of narrative-only labels, set of subnarrative-only labels)
    The labels in each column are expected to be separated by semicolons.
    """
    data = {}
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue  
            article_id = parts[0].strip()
            narratives = [lab.strip() for lab in parts[1].split(";") if lab.strip()]
            subnarratives = [lab.strip() for lab in parts[2].split(";") if lab.strip()]
            pairs = set(zip(narratives, subnarratives)) 
            narrative_only = set(narratives) 
            subnarrative_only = set(subnarratives) 
            data[article_id] = (pairs, narrative_only, subnarrative_only)
    return data

def evaluate_files(gold_file, pred_file):
    """
    Reads the gold and prediction text files, computes:
      - The averaged sample F1 scores for:
            * full narrative:subnarrative pairs,
            * narrative-only labels,
            * subnarrative-only labels.
      - The macro F1 scores for:
            * narrative-only labels, and
            * subnarrative-only labels.
    Returns:
      (avg_f1_pairs, avg_f1_narrative, avg_f1_subnarrative, macro_f1_narrative, macro_f1_subnarrative)
    """
    gold_data = read_txt_labels(gold_file)
    pred_data = read_txt_labels(pred_file)
    
    # Use the union of all article_ids found in gold and predictions.
    all_article_ids = set(gold_data.keys()).union(set(pred_data.keys()))
    
    f1_pairs_scores = []
    f1_narrative_scores = []
    f1_subnarrative_scores = []
    
    for article_id in all_article_ids:
        # Get gold labels; if missing, treat as empty.
        gold_pairs, gold_narrative, gold_subnarrative = gold_data.get(article_id, (set(), set(), set()))
        pred_pairs, pred_narrative, pred_subnarrative = pred_data.get(article_id, (set(), set(), set()))
        
        sample_f1_pairs = f1_for_sample(gold_pairs, pred_pairs)
        sample_f1_narrative = f1_for_sample(gold_narrative, pred_narrative)
        sample_f1_subnarrative = f1_for_sample(gold_subnarrative, pred_subnarrative)
        
        f1_pairs_scores.append(sample_f1_pairs)
        f1_narrative_scores.append(sample_f1_narrative)
        f1_subnarrative_scores.append(sample_f1_subnarrative)
    
    avg_f1_pairs = sum(f1_pairs_scores) / len(f1_pairs_scores) if f1_pairs_scores else 0.0
    avg_f1_narrative = sum(f1_narrative_scores) / len(f1_narrative_scores) if f1_narrative_scores else 0.0
    avg_f1_subnarrative = sum(f1_subnarrative_scores) / len(f1_subnarrative_scores) if f1_subnarrative_scores else 0.0

    # Build the universal label sets for narrative-only and subnarrative-only.
    all_narrative_labels = set()
    all_subnarrative_labels = set()
    for article_id, (gold_pairs, gold_narrative, gold_subnarrative) in gold_data.items():
        all_narrative_labels.update(gold_narrative)
        all_subnarrative_labels.update(gold_subnarrative)
    for article_id, (pred_pairs, pred_narrative, pred_subnarrative) in pred_data.items():
        all_narrative_labels.update(pred_narrative)
        all_subnarrative_labels.update(pred_subnarrative)
    
    # Sort the labels for consistent ordering.
    all_narrative_labels = sorted(all_narrative_labels)
    all_subnarrative_labels = sorted(all_subnarrative_labels)
    
    # Build binary indicator vectors for each article.
    y_true_narrative = []
    y_pred_narrative = []
    y_true_subnarrative = []
    y_pred_subnarrative = []
    
    for article_id in all_article_ids:
        gold_pairs, gold_narrative, gold_subnarrative = gold_data.get(article_id, (set(), set(), set()))
        pred_pairs, pred_narrative, pred_subnarrative = pred_data.get(article_id, (set(), set(), set()))
        
        # For narrative-only labels.
        gold_vector_narrative = [1 if label in gold_narrative else 0 for label in all_narrative_labels]
        pred_vector_narrative = [1 if label in pred_narrative else 0 for label in all_narrative_labels]
        y_true_narrative.append(gold_vector_narrative)
        y_pred_narrative.append(pred_vector_narrative)
        
        # For subnarrative-only labels.
        gold_vector_subnarrative = [1 if label in gold_subnarrative else 0 for label in all_subnarrative_labels]
        pred_vector_subnarrative = [1 if label in pred_subnarrative else 0 for label in all_subnarrative_labels]
        y_true_subnarrative.append(gold_vector_subnarrative)
        y_pred_subnarrative.append(pred_vector_subnarrative)
    
    # Compute macro F1 scores using sklearn's f1_score.
    macro_f1_narrative = f1_score(y_true_narrative, y_pred_narrative, average='macro', zero_division=0)
    macro_f1_subnarrative = f1_score(y_true_subnarrative, y_pred_subnarrative, average='macro', zero_division=0)
    
    print("Averaged sample F1 (narrative:subnarrative pairs): {:.4f}".format(avg_f1_pairs))
    print("Averaged sample F1 (narrative only): {:.4f}".format(avg_f1_narrative))
    print("Averaged sample F1 (subnarrative only): {:.4f}".format(avg_f1_subnarrative))
    print("Macro F1 (narrative only): {:.4f}".format(macro_f1_narrative))
    print("Macro F1 (subnarrative only): {:.4f}".format(macro_f1_subnarrative))
    
    return avg_f1_pairs, avg_f1_narrative, avg_f1_subnarrative, macro_f1_narrative, macro_f1_subnarrative

if __name__ == "__main__":
    gold_file = os.path.join("data", "annotations", "annotation.txt")
    pred_file = os.path.join("outputs", "submission.txt")
    evaluate_files(gold_file, pred_file)
