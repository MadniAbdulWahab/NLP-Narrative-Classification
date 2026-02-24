import os
import json
import logging
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------------------
# Helper functions
# ---------------------------
def compute_label_mapping(df, column, delimiter=";"):
    labels = set()
    for entry in df[column]:
        for lab in str(entry).split(delimiter):
            lab = lab.strip()
            if lab:
                labels.add(lab)
    labels = sorted(list(labels))
    mapping = {label: idx for idx, label in enumerate(labels)}
    return mapping

def compute_pos_weights(df, column, label2id, delimiter=";"):
    counts = np.zeros(len(label2id))
    N = len(df)
    for entry in df[column]:
        labs = [lab.strip() for lab in str(entry).split(delimiter)]
        for lab in labs:
            if lab in label2id:
                counts[label2id[lab]] += 1
    pos_weights = []
    for i in range(len(label2id)):
        if counts[i] > 0:
            pos_weight = (N - counts[i]) / counts[i]
        else:
            pos_weight = 1.0
        pos_weights.append(pos_weight)
    return torch.tensor(pos_weights, dtype=torch.float)

def compute_sample_weights(df, column, label2id, delimiter=";"):
    counts = np.zeros(len(label2id))
    for entry in df[column]:
        labs = [lab.strip() for lab in str(entry).split(delimiter) if lab.strip()]
        for lab in labs:
            if lab in label2id:
                counts[label2id[lab]] += 1
    sample_weights = []
    for entry in df[column]:
        labs = [lab.strip() for lab in str(entry).split(delimiter) if lab.strip()]
        if len(labs) == 0:
            sample_weights.append(1.0)
        else:
            weights = [1.0 / counts[label2id[lab]] for lab in labs if counts[label2id[lab]] > 0]
            sample_weights.append(np.mean(weights))
    return sample_weights

# ---------------------------
# Simple text augmentation: Randomly drop words.
# ---------------------------
def augment_text(text, drop_prob=0.05): 
    words = text.split()
    if not words:
        return text
    new_words = [word for word in words if random.random() > drop_prob]
    return " ".join(new_words) if new_words else text

# ---------------------------
# Focal Loss
# ---------------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, pos_weight=None, reduction="mean"): 
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, pos_weight=self.pos_weight, reduction='none'
        )
        pt = torch.exp(-bce_loss)
        loss = (1 - pt) ** self.gamma * bce_loss
        if self.alpha is not None:
            loss = self.alpha * loss
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

# ---------------------------
# Label selection function with optional forced top prediction
# ---------------------------
def select_labels(probabilities, primary_threshold=0.7, fallback_lower=0.65, force_top=True):
    """
    Given a tensor of probabilities, return a binary prediction list.
    First, select any label with probability above primary_threshold.
    If none are selected and force_top is True, then force the top prediction
    only if its probability is above fallback_lower.
    """
    preds = (probabilities > primary_threshold).int().tolist()
    if sum(preds) == 0 and force_top:
        sorted_indices = torch.argsort(probabilities, descending=True)
        if probabilities[sorted_indices[0]].item() > fallback_lower:
            preds[sorted_indices[0]] = 1
        logging.debug("Fallback selection applied: selected index %s with probability %.4f.",
                      sorted_indices[0].item(), probabilities[sorted_indices[0]].item())
    return preds

# ---------------------------
# Dataset Class
# ---------------------------
class NarrativeDataset(Dataset):
    def __init__(self, dataframe, articles_dir, tokenizer, max_length, 
                 narrative_label2id, subnarrative_label2id, augment=False, drop_prob=0.05):
        self.data = dataframe.reset_index(drop=True)
        self.articles_dir = articles_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.narrative_label2id = narrative_label2id
        self.subnarrative_label2id = subnarrative_label2id
        self.augment = augment
        self.drop_prob = drop_prob
        logging.info(f"Dataset initialized with {len(self.data)} samples. Augmentation: {self.augment}")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        article_id = row["article_id"]
        narrative_labels = [lab.strip() for lab in str(row["narrative"]).split(";")]
        subnarrative_labels = [lab.strip() for lab in str(row["subnarrative"]).split(";")]
        
        article_path = os.path.join(self.articles_dir, article_id)
        try:
            with open(article_path, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception as e:
            logging.error(f"Error reading file {article_path}: {e}")
            text = ""
        
        if self.augment:
            text = augment_text(text, drop_prob=self.drop_prob)
            
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        narrative_vector = torch.zeros(len(self.narrative_label2id), dtype=torch.float)
        subnarrative_vector = torch.zeros(len(self.subnarrative_label2id), dtype=torch.float)
        for lab in narrative_labels:
            if lab in self.narrative_label2id:
                narrative_vector[self.narrative_label2id[lab]] = 1.0
        for lab in subnarrative_labels:
            if lab in self.subnarrative_label2id:
                subnarrative_vector[self.subnarrative_label2id[lab]] = 1.0
                
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "narrative_labels": narrative_vector,
            "subnarrative_labels": subnarrative_vector,
        }

# ---------------------------
# Model: Multi-label classifier with hierarchical conditioning and consistency loss.
# ---------------------------
class MultiLabelClassifier(nn.Module):
    def __init__(self, model_name, num_narrative_labels, num_subnarrative_labels,
                 pos_weight_narrative=None, pos_weight_subnarrative=None,
                 use_focal_loss=False, gamma=2.0, alpha_narrative=0.5, alpha_subnarrative=0.5,
                 hierarchical_mapping=None, narrative_id2label=None, subnarrative_id2label=None,
                 consistency_weight=0.5, subnarrative_loss_weight=1.0):
        """
        subnarrative_loss_weight: multiplier for subnarrative loss (default 1.0; adjust as needed)
        """
        super(MultiLabelClassifier, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.2)
        self.use_focal_loss = use_focal_loss
        self.gamma = gamma
        self.alpha_narrative = alpha_narrative
        self.alpha_subnarrative = alpha_subnarrative
        self.pos_weight_narrative = pos_weight_narrative
        self.pos_weight_subnarrative = pos_weight_subnarrative
        
        # Save hierarchical mapping and label dictionaries for consistency loss.
        self.hierarchical_mapping = hierarchical_mapping
        self.narrative_id2label = narrative_id2label
        self.subnarrative_id2label = subnarrative_id2label
        self.consistency_weight = consistency_weight
        self.subnarrative_loss_weight = subnarrative_loss_weight
        
        self.narrative_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, num_narrative_labels)
        )
        
        self.subnarrative_head = nn.Sequential(
            nn.Linear(hidden_size + num_narrative_labels, hidden_size),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, num_subnarrative_labels)
        )
        
        logging.info("Model initialized with hierarchical conditioning. Use focal loss: %s", self.use_focal_loss)
        
    def forward(self, input_ids, attention_mask, narrative_labels=None, subnarrative_labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        else:
            pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        narrative_logits = self.narrative_head(pooled_output)
        sub_input = torch.cat([pooled_output, narrative_logits], dim=1)
        subnarrative_logits = self.subnarrative_head(sub_input)
        
        loss = None
        if narrative_labels is not None and subnarrative_labels is not None:
            narrative_labels = narrative_labels.to(narrative_logits.device)
            subnarrative_labels = subnarrative_labels.to(subnarrative_logits.device)
            if self.use_focal_loss:
                focal_loss_narrative = FocalLoss(
                    gamma=self.gamma, alpha=self.alpha_narrative,
                    pos_weight=self.pos_weight_narrative.to(narrative_logits.device) if self.pos_weight_narrative is not None else None,
                    reduction="mean"
                )
                focal_loss_subnarrative = FocalLoss(
                    gamma=self.gamma, alpha=self.alpha_subnarrative,
                    pos_weight=self.pos_weight_subnarrative.to(subnarrative_logits.device) if self.pos_weight_subnarrative is not None else None,
                    reduction="mean"
                )
                narrative_loss = focal_loss_narrative(narrative_logits, narrative_labels)
                subnarrative_loss = focal_loss_subnarrative(subnarrative_logits, subnarrative_labels)
            else:
                loss_fct_narrative = nn.BCEWithLogitsLoss(
                    pos_weight=self.pos_weight_narrative.to(narrative_logits.device) if self.pos_weight_narrative is not None else None
                )
                loss_fct_subnarrative = nn.BCEWithLogitsLoss(
                    pos_weight=self.pos_weight_subnarrative.to(subnarrative_logits.device) if self.pos_weight_subnarrative is not None else None
                )
                narrative_loss = loss_fct_narrative(narrative_logits, narrative_labels)
                subnarrative_loss = loss_fct_subnarrative(subnarrative_logits, subnarrative_labels)
            
            # Adjusted loss weighting: use a configurable multiplier for subnarrative loss.
            total_loss = narrative_loss + self.subnarrative_loss_weight * subnarrative_loss
            
            # Add hierarchical consistency loss:
            if self.hierarchical_mapping is not None and self.narrative_id2label is not None:
                sub_probs = torch.sigmoid(subnarrative_logits)
                consistency_loss = 0.0
                count = 0
                for sample_idx in range(narrative_labels.shape[0]):
                    for narr_idx in range(narrative_labels.shape[1]):
                        if narrative_labels[sample_idx, narr_idx] == 1:
                            if self.narrative_id2label.get(narr_idx, "").lower() == "other":
                                continue
                            group = self.hierarchical_mapping.get(narr_idx, [])
                            if group:
                                group_probs = sub_probs[sample_idx, group]
                                max_prob = torch.max(group_probs)
                                consistency_loss += (1 - max_prob)
                                count += 1
                if count > 0:
                    consistency_loss = consistency_loss / count
                else:
                    consistency_loss = 0.0
                total_loss = total_loss + (self.consistency_weight * consistency_loss)
            
            loss = total_loss
        
        return {
            "loss": loss,
            "narrative_logits": narrative_logits,
            "subnarrative_logits": subnarrative_logits,
        }

# ---------------------------
# Data collator
# ---------------------------
def collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    narrative_labels = torch.stack([item["narrative_labels"] for item in batch])
    subnarrative_labels = torch.stack([item["subnarrative_labels"] for item in batch])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "narrative_labels": narrative_labels,
        "subnarrative_labels": subnarrative_labels,
    }

# ---------------------------
# Compute Metrics Function with updated thresholds
# ---------------------------
def compute_metrics_subtask2(eval_pred, threshold_narrative=0.7, threshold_subnarrative=0.7, 
                              narrative_id2label=None, subnarrative_id2label=None):
    predictions = eval_pred.predictions
    label_ids = eval_pred.label_ids

    logits_narrative = predictions["narrative_logits"]
    logits_subnarrative = predictions["subnarrative_logits"]
    labels_narrative = np.array(label_ids["narrative_labels"])
    labels_subnarrative = np.array(label_ids["subnarrative_labels"])
    
    preds_narrative = (torch.sigmoid(torch.tensor(logits_narrative)) > threshold_narrative).int().numpy()
    preds_subnarrative = (torch.sigmoid(torch.tensor(logits_subnarrative)) > threshold_subnarrative).int().numpy()
    
    f1_narrative = f1_score(labels_narrative, preds_narrative, average="samples", zero_division=0)
    f1_subnarrative = f1_score(labels_subnarrative, preds_subnarrative, average="samples", zero_division=0)
    macro_f1_narrative = f1_score(labels_narrative, preds_narrative, average="macro", zero_division=0)
    macro_f1_subnarrative = f1_score(labels_subnarrative, preds_subnarrative, average="macro", zero_division=0)
    
    metrics = {
        "f1_narrative": f1_narrative,
        "f1_subnarrative": f1_subnarrative,
        "macro_f1_narrative": macro_f1_narrative,
        "macro_f1_subnarrative": macro_f1_subnarrative,
    }
    
    prec_narrative, recall_narrative, f1_narrative_arr, _ = precision_recall_fscore_support(
        labels_narrative, preds_narrative, average=None, zero_division=0)
    prec_subnarrative, recall_subnarrative, f1_subnarrative_arr, _ = precision_recall_fscore_support(
        labels_subnarrative, preds_subnarrative, average=None, zero_division=0)
    
    if narrative_id2label is not None:
        per_label_narrative = {narrative_id2label[i]: {"precision": prec_narrative[i],
                                                       "recall": recall_narrative[i],
                                                       "f1": f1_narrative_arr[i]}
                                for i in range(len(prec_narrative))}
        metrics["per_label_narrative"] = per_label_narrative
        logging.info(f"Per-label narrative metrics: {per_label_narrative}")
    if subnarrative_id2label is not None:
        per_label_subnarrative = {subnarrative_id2label[i]: {"precision": prec_subnarrative[i],
                                                             "recall": recall_subnarrative[i],
                                                             "f1": f1_subnarrative_arr[i]}
                                  for i in range(len(prec_subnarrative))}
        metrics["per_label_subnarrative"] = per_label_subnarrative
        logging.info(f"Per-label subnarrative metrics: {per_label_subnarrative}")
    
    logging.info(f"Evaluation metrics computed: {metrics}")
    return metrics

# ---------------------------
# Custom Trainer with WeightedRandomSampler for oversampling
# ---------------------------

class CustomTrainerWithSampler(Trainer):
    def __init__(self, *args, sample_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_weights = sample_weights

    def get_train_dataloader(self):
        if self.sample_weights is None:
            return super().get_train_dataloader()
        else:
            sampler = WeightedRandomSampler(self.sample_weights, num_samples=len(self.sample_weights), replacement=True)
            return DataLoader(
                self.train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                sampler=sampler,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
            )
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            narrative_labels=inputs["narrative_labels"],
            subnarrative_labels=inputs["subnarrative_labels"],
        )
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                narrative_labels=inputs["narrative_labels"],
                subnarrative_labels=inputs["subnarrative_labels"],
            )
        loss = outputs["loss"]
        narrative_logits = outputs["narrative_logits"].detach().cpu()
        subnarrative_logits = outputs["subnarrative_logits"].detach().cpu()
        labels = {
            "narrative_labels": inputs["narrative_labels"],
            "subnarrative_labels": inputs["subnarrative_labels"],
        }
        return (loss, {"narrative_logits": narrative_logits, "subnarrative_logits": subnarrative_logits}, labels)

# ---------------------------
# Main training function
# ---------------------------
def main():
    logging.info("Starting training script.")
    
    model_name = "bert-base-multilingual-cased"
    max_length = 512
    articles_dir = os.path.join("data", "articles")
    annotations_file = os.path.join("data", "annotations", "annotation.txt")
    
    df = pd.read_csv(annotations_file, sep="\t", header=None, names=["article_id", "narrative", "subnarrative"])
    logging.info(f"Annotation file read. Total samples: {len(df)}")
    
    narrative_label2id = compute_label_mapping(df, "narrative", delimiter=";")
    subnarrative_label2id = compute_label_mapping(df, "subnarrative", delimiter=";")
    logging.info(f"Narrative labels mapping: {narrative_label2id}")
    logging.info(f"Subnarrative labels mapping: {subnarrative_label2id}")
    
    narrative_id2label = {v: k for k, v in narrative_label2id.items()}
    subnarrative_id2label = {v: k for k, v in subnarrative_label2id.items()}
    
    model_dir = os.path.join("models", "final_model")
    os.makedirs(model_dir, exist_ok=True)

    # write mappings
    with open(os.path.join(model_dir, "narrative_mapping.json"), "w", encoding="utf-8") as f:
        json.dump(narrative_label2id, f, ensure_ascii=False, indent=2)
    with open(os.path.join(model_dir, "subnarrative_mapping.json"), "w", encoding="utf-8") as f:
        json.dump(subnarrative_label2id, f, ensure_ascii=False, indent=2)
        
        
    num_narrative_labels = len(narrative_label2id)
    num_subnarrative_labels = len(subnarrative_label2id)
    
    hierarchical_mapping = {}
    for narr_label, narr_id in narrative_label2id.items():
        if narr_label.lower() == "other":
            continue
        group = []
        prefix = narr_label + ":"
        for sub_label, sub_id in subnarrative_label2id.items():
            if sub_label.startswith(prefix):
                group.append(sub_id)
        if group:
            hierarchical_mapping[narr_id] = group
    logging.info(f"Hierarchical mapping: {hierarchical_mapping}")
    
    train_df, dev_df = train_test_split(df, test_size=0.1, random_state=42)
    logging.info(f"Train samples: {len(train_df)}, Dev samples: {len(dev_df)}")
    
    pos_weight_narrative = compute_pos_weights(train_df, "narrative", narrative_label2id, delimiter=";")
    pos_weight_subnarrative = compute_pos_weights(train_df, "subnarrative", subnarrative_label2id, delimiter=";")
    logging.info(f"Computed pos_weight for narrative: {pos_weight_narrative.tolist()}")
    logging.info(f"Computed pos_weight for subnarrative: {pos_weight_subnarrative.tolist()}")
    
    sample_weights = compute_sample_weights(train_df, "narrative", narrative_label2id, delimiter=";")
    
    logging.info("Creating training dataset with augmentation...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset = NarrativeDataset(
        dataframe=train_df,
        articles_dir=articles_dir,
        tokenizer=tokenizer,
        max_length=max_length,
        narrative_label2id=narrative_label2id,
        subnarrative_label2id=subnarrative_label2id,
        augment=True,
        drop_prob=0.05  
    )
    
    logging.info("Creating development dataset (no augmentation)...")
    dev_dataset = NarrativeDataset(
        dataframe=dev_df,
        articles_dir=articles_dir,
        tokenizer=tokenizer,
        max_length=max_length,
        narrative_label2id=narrative_label2id,
        subnarrative_label2id=subnarrative_label2id,
        augment=False
    )
    
    logging.info("Loading model...")
    model = MultiLabelClassifier(
        model_name,
        num_narrative_labels,
        num_subnarrative_labels,
        pos_weight_narrative=pos_weight_narrative,
        pos_weight_subnarrative=pos_weight_subnarrative,
        use_focal_loss=True,
        gamma=1.5, 
        alpha_narrative=0.5,
        alpha_subnarrative=0.5,
        hierarchical_mapping=hierarchical_mapping,
        narrative_id2label=narrative_id2label,
        subnarrative_id2label=subnarrative_id2label,
        consistency_weight=0.7,  
        subnarrative_loss_weight=1.0
    )
    
    training_args = TrainingArguments(
        output_dir="./output",
        num_train_epochs=30,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        learning_rate=1e-5, 
        weight_decay=0.01,
        warmup_steps=500,
        max_grad_norm=1.0,
        load_best_model_at_end=True,
        metric_for_best_model="f1_subnarrative",
        lr_scheduler_type="cosine"
    )
    logging.info("Training arguments defined.")
    
    trainer = CustomTrainerWithSampler(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=collate_fn,
        sample_weights=sample_weights,
        compute_metrics=lambda eval_pred: compute_metrics_subtask2(
            eval_pred,
            threshold_narrative=0.7,  
            threshold_subnarrative=0.7,  
            narrative_id2label=narrative_id2label,
            subnarrative_id2label=subnarrative_id2label
        )
    )
    
    logging.info("Trainer initialized. Starting training...")
    trainer.train()
    logging.info("Training completed.")
    
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)
    model.encoder.config.save_pretrained(model_dir)
    logging.info("Final model, tokenizer, and config saved to './final_model'.")

if __name__ == "__main__":
    main()
