import os
import json
import logging
import torch
from transformers import AutoTokenizer
from training import MultiLabelClassifier  

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def load_mappings(mapping_dir="./final_model"):
    """
    Load the narrative and subnarrative mappings and build inverse mappings.
    """
    logging.debug("Loading label mappings from '%s'.", mapping_dir)
    with open(os.path.join(mapping_dir, "narrative_mapping.json"), "r", encoding="utf-8") as f:
        narrative_label2id = json.load(f)
    with open(os.path.join(mapping_dir, "subnarrative_mapping.json"), "r", encoding="utf-8") as f:
        subnarrative_label2id = json.load(f)
    narrative_id2label = {int(v): k for k, v in narrative_label2id.items()}
    subnarrative_id2label = {int(v): k for k, v in subnarrative_label2id.items()}
    logging.debug("Narrative mapping loaded: %s", narrative_id2label)
    logging.debug("Subnarrative mapping loaded: %s", subnarrative_id2label)
    return narrative_id2label, subnarrative_id2label

def load_model(mapping_dir="./final_model", device="cpu"):
    """
    Load the trained model weights.
    This function first tries to load safetensors weights.
    If not found, it loads the standard PyTorch weights (pytorch_model.bin).
    """
    logging.debug("Loading model from '%s' on device '%s'.", mapping_dir, device)
    narrative_id2label, subnarrative_id2label = load_mappings(mapping_dir)
    num_narrative_labels = len(narrative_id2label)
    num_subnarrative_labels = len(subnarrative_id2label)
    model_name = "bert-base-multilingual-cased"
    
    # Initialize the model.
    model = MultiLabelClassifier(model_name, num_narrative_labels, num_subnarrative_labels)
    logging.debug("Initialized model with %d narrative labels and %d subnarrative labels.",
                  num_narrative_labels, num_subnarrative_labels)
    
    # Define paths for weights.
    safetensor_path = os.path.join(mapping_dir, "model.safetensors")
    pt_model_path = os.path.join(mapping_dir, "pytorch_model.bin")
    
    if os.path.exists(safetensor_path):
        from safetensors.torch import load_file
        state_dict = load_file(safetensor_path)
        logging.info("Loaded model weights from safetensors.")
    elif os.path.exists(pt_model_path):
        state_dict = torch.load(pt_model_path, map_location=device)
        logging.info("Loaded model weights from pytorch_model.bin.")
    else:
        logging.error("No model weight file found in '%s'!", mapping_dir)
        raise FileNotFoundError("Model weight file not found.")
    
    model.load_state_dict(state_dict)
    logging.debug("Model state_dict keys: %s", list(state_dict.keys()))
    return model.to(device), narrative_id2label, subnarrative_id2label

def enforce_hierarchical_consistency(narrative_labels, subnarrative_labels):
    """
    Enforce the Subtask 2 convention:
      - If no narrative is predicted or the only narrative is "Other", then set subnarrative to ["Other"].
      - Otherwise, for each predicted narrative (except "Other"), if no subnarrative label 
        starting with that narrative (i.e. "Narrative:") is present, append the default label "[Narrative]: Other".
    """
    logging.debug("Before hierarchical enforcement, narrative_labels: %s, subnarrative_labels: %s",
                  narrative_labels, subnarrative_labels)
    if not narrative_labels or (len(narrative_labels) == 1 and narrative_labels[0] == "Other"):
        logging.debug("No valid narrative labels predicted; setting subnarrative to ['Other'].")
        return ["Other"]
    
    updated_subnarrative_labels = subnarrative_labels.copy()
    for narr in narrative_labels:
        if narr != "Other":
            prefix = narr + ":"
            if not any(sub.startswith(prefix) for sub in subnarrative_labels):
                default_sub = narr + ": Other"
                logging.debug("No subnarrative label found for '%s'; appending default label '%s'.", narr, default_sub)
                updated_subnarrative_labels.append(default_sub)
    logging.debug("After hierarchical enforcement, subnarrative_labels: %s", updated_subnarrative_labels)
    return updated_subnarrative_labels

def select_labels(probabilities, primary_threshold=0.30, fallback_lower=0.25, top_k=2):
    """
    Given a tensor of probabilities, select labels as follows:
      1. Select any label with probability above primary_threshold.
      2. If none are selected, check the top_k labels and add those with probability above fallback_lower.
      3. Return a binary prediction list.
    """
    preds = (probabilities > primary_threshold).int().tolist()
    if sum(preds) == 0:
        sorted_indices = torch.argsort(probabilities, descending=True)
        # Always force the top prediction.
        preds[sorted_indices[0]] = 1
        # Also consider adding a second label if its probability is above the fallback_lower threshold.
        if len(probabilities) > 1 and probabilities[sorted_indices[1]].item() > fallback_lower:
            preds[sorted_indices[1]] = 1
        logging.debug("Fallback selection applied: selected indices %s with probabilities %s.",
                      sorted_indices[:top_k].tolist(), probabilities[sorted_indices[:top_k]].tolist())
    return preds

def run_inference(model, tokenizer, folder_path, narrative_id2label, subnarrative_id2label,
                  max_length=512, primary_threshold_narrative=0.30, primary_threshold_subnarrative=0.30,
                  fallback_lower_narrative=0.25, fallback_lower_subnarrative=0.25, device="cpu"):
    """
    Process each article file in the folder:
      - Tokenize and run through the model.
      - Apply sigmoid to obtain probabilities.
      - Use a dynamic selection strategy:
          * First, select labels with probability above the primary threshold.
          * If none pass, force the top (or top_k) predictions if above a fallback lower bound.
      - Log detailed probability values and indices.
      - Enforce hierarchical consistency.
    Returns a list of tuples: (article_id, narrative_labels, subnarrative_labels).
    """
    logging.info("Starting inference on folder: '%s'", folder_path)
    model.eval()
    predictions = []
    file_list = sorted(os.listdir(folder_path))
    logging.info("Found %d files for inference.", len(file_list))
    
    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            logging.debug("Read file '%s' (length: %d characters).", file_name, len(text))
        except Exception as e:
            logging.error("Error reading %s: %s", file_path, e)
            continue
        
        # Tokenize the text.
        encoding = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        logging.debug("Tokenization complete for '%s'. Input shape: %s", file_name, input_ids.shape)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Obtain raw logits.
        narrative_logits = outputs["narrative_logits"].squeeze()
        subnarrative_logits = outputs["subnarrative_logits"].squeeze()
        logging.debug("Raw narrative logits for '%s': %s", file_name, narrative_logits[:5].tolist())
        logging.debug("Raw subnarrative logits for '%s': %s", file_name, subnarrative_logits[:5].tolist())
        
        # Compute probabilities.
        narrative_probs = torch.sigmoid(narrative_logits)
        subnarrative_probs = torch.sigmoid(subnarrative_logits)
        logging.debug("Narrative probabilities for '%s': %s", file_name, narrative_probs.tolist())
        logging.debug("Subnarrative probabilities for '%s': %s", file_name, subnarrative_probs.tolist())
        
        # Select predictions using our dynamic selection.
        narrative_preds = select_labels(narrative_probs, primary_threshold=primary_threshold_narrative, fallback_lower=fallback_lower_narrative)
        subnarrative_preds = select_labels(subnarrative_probs, primary_threshold=primary_threshold_subnarrative, fallback_lower=fallback_lower_subnarrative)
        logging.debug("Dynamic predictions for '%s' - Narrative: %s, Subnarrative: %s",
                      file_name, narrative_preds, subnarrative_preds)
        
        narrative_labels = [narrative_id2label[i] for i, pred in enumerate(narrative_preds) if pred == 1]
        subnarrative_labels = [subnarrative_id2label[i] for i, pred in enumerate(subnarrative_preds) if pred == 1]
        logging.debug("Initial labels for '%s' - Narrative: %s, Subnarrative: %s",
                      file_name, narrative_labels, subnarrative_labels)
        
        # Enforce hierarchical consistency.
        subnarrative_labels = enforce_hierarchical_consistency(narrative_labels, subnarrative_labels)
        logging.debug("Final subnarrative labels after hierarchical enforcement for '%s': %s",
                      file_name, subnarrative_labels)
        
        if len(narrative_labels) == 1 and narrative_labels[0] == "Other":
            logging.warning("For '%s', only 'Other' is predicted for narrative.", file_name)
        
        predictions.append((file_name, ";".join(narrative_labels), ";".join(subnarrative_labels)))
        logging.info("Processed %s: Narrative -> %s; Subnarrative -> %s", 
                     file_name, narrative_labels, subnarrative_labels)
    
    return predictions

def main():
    logging.info("Starting inference on development set for submission.")
    dev_folder = os.path.join("data", "validation")
    mapping_dir = os.path.join("models", "final_model")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logging.debug("Loading tokenizer from '%s'.", mapping_dir)
    tokenizer = AutoTokenizer.from_pretrained(mapping_dir)
    model, narrative_id2label, subnarrative_id2label = load_model(mapping_dir, device=device)
    
    predictions = run_inference(
        model, tokenizer, dev_folder,
        narrative_id2label, subnarrative_id2label,
        max_length=512,
        primary_threshold_narrative=0.60,
        primary_threshold_subnarrative=0.75,
        fallback_lower_narrative=0.40,
        fallback_lower_subnarrative=0.40,
        device=device
    )
    
    output_file = os.path.join("outputs", "submission.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        for article_id, narrative, subnarrative in predictions:
            f.write(f"{article_id}\t{narrative}\t{subnarrative}\n")
    logging.info("Submission file saved to '%s'.", output_file)

if __name__ == "__main__":
    main()
