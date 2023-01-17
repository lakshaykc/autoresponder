from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
import evaluate
import nltk
import numpy as np
from nltk.tokenize import sent_tokenize
from datasets import load_from_disk

nltk.download("punkt")

# Load the dataset from the ./data/chat_history folder
ds = load_from_disk("data/chat_history")

model_id = "google/flan-t5-xl"
model_dir = "models/t5-xl-20ep"

# Metric
metric = evaluate.load("rouge")

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

training_batch_size = 4

# helper functions to postprocess text. Adapted from https://www.philschmid.de/fine-tune-flan-t5
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(sent_tokenize(label)) for label in labels]

    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    result = {k: round(v * 100, 4) for k, v in result.items()}
    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
    ]
    result["gen_len"] = np.mean(prediction_lens)
    return result


training_args = Seq2SeqTrainingArguments(
    output_dir=model_dir,
    per_device_train_batch_size=training_batch_size,
    per_device_eval_batch_size=training_batch_size,
    predict_with_generate=True,
    fp16=False,
    # Learning rate recommended by https://huggingface.co/docs/transformers/model_doc/t5
    learning_rate=1e-4,
    num_train_epochs=5,
    # logging & evaluation strategies
    logging_dir=f"{model_dir}/logs",
    logging_strategy="steps",
    logging_steps=2000,
    evaluation_strategy="steps",
    save_strategy="steps",
    save_steps=2000,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="rouge1",
    report_to="tensorboard",
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
    compute_metrics=compute_metrics,
)

# https://discuss.huggingface.co/t/how-to-evaluate-before-first-training-step/18838
trainer.evaluate()

trainer.train()
