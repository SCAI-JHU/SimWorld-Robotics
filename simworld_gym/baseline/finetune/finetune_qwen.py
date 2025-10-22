import json
import os
import datasets
import random
import torch
from transformers import BitsAndBytesConfig
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from qwen_vl_utils import process_vision_info
import wandb
from tqdm import tqdm
from trl import SFTTrainer, SFTConfig
from prompt_template import oneshot_prompt


random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=8,
    bias="none",
    target_modules=["q_proj", "v_proj", "visual.merger.mlp.0", "visual.merger.mlp.2"],
    task_type="CAUSAL_LM",
)

train_dataset = datasets.load_dataset("Jise/citynav", split="train")
eval_dataset = datasets.load_dataset("Jise/citynav", split="validation")
model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
)
model = prepare_model_for_kbit_training(model)
peft_model = get_peft_model(model, peft_config)
# for name, param in peft_model.named_parameters():
#     if "visual.merger.ln_q" in name:
#         print(f"Unfreezing LayerNorm: {name}")
#         param.requires_grad = True
peft_model.print_trainable_parameters()

processor = Qwen2_5_VLProcessor.from_pretrained(model_id)
processor.tokenizer.add_special_tokens({
    "additional_special_tokens": ["<OBS>", "<EOBS>", "<TAR>", "<ETAR>"]
})

label2id = train_dataset.features["action"].str2int

action_mapping = {
    label2id("Move forward"): 0,
    label2id("Turn left"): 1,
    label2id("Turn right"): 2,
    label2id("Subtask completed"): -1,
}
    
def format_data(sample):
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": oneshot_prompt}]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "<OBS>"},
                {"type": "image", "image": sample["current_view"]},
                {"type": "text", "text": "<EOBS><TAR>"},
                {"type": "image", "image": sample["expected_view"]},
                {"type": "text", "text": "<ETAR>"},
                {"type": "text", "text": f"Current orientation: {sample['orientation']}\n"},
                {"type": "text", "text": f"Current subtask: {sample['subtask']}\n"},
                {"type": "text", "text": f"Action history: {sample['history']}\n"},
                {"type": "text", "text": "Output the estimated final orientation, distance and plan for remaining actions and the next action."}
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": f"```json\n{json.dumps({
                    "Expected_Orientation": sample["target_orientation"], 
                    "Remaining_Distance": sample["distance"],
                    "Remaining_Actions": sample["plan"],
                    "Next_Action": action_mapping[sample["action"]]
                }, indent=2)}\n```"},
            ]
        }
    ]


print("Applying chat template to dataset...")
train_dataset = [format_data(sample) for sample in tqdm(train_dataset)]
eval_dataset = [format_data(sample) for sample in eval_dataset]

def find_subsequence(tokens, subseq):
    for i in range(len(tokens) - len(subseq) + 1):
        if tokens[i:i + len(subseq)] == subseq:
            return i
    return -1

def collate_fn(examples):
    texts = [
        processor.apply_chat_template(example, tokenize=False) for example in examples
    ]
    image_inputs = [process_vision_info(example)[0] for example in examples]

    batch = processor(
        text=texts, images=image_inputs, return_tensors="pt", padding=True
    )

    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    image_token_ids = [151652, 151653, 151655]
    for image_token_id in image_token_ids:
        labels[labels == image_token_id] = -100

    assistant_token_ids = [151644, 77091]
    for i, input_id in enumerate(batch["input_ids"]):
        input_id_list = input_id.tolist()
        start = find_subsequence(input_id_list, assistant_token_ids)
        if start != -1:
            labels[i, :start + len(assistant_token_ids)] = -100

    batch["labels"] = labels
    return batch

print(processor.apply_chat_template(train_dataset[0], tokenize=False))

# Configure training arguments
training_args = SFTConfig(
    output_dir="qwen2.5-7b-instruct-citynav",  # Directory to save the model
    num_train_epochs=2,  # Number of training epochs
    per_device_train_batch_size=4,  # Batch size for training
    per_device_eval_batch_size=4,  # Batch size for evaluation
    gradient_accumulation_steps=8,  # Steps to accumulate gradients
    gradient_checkpointing=True,  # Enable gradient checkpointing for memory efficiency
    # Optimizer and scheduler settings
    optim="adamw_torch_fused",  # Optimizer type
    learning_rate=2e-4,  # Learning rate for training
    lr_scheduler_type="constant",  # Type of learning rate scheduler
    # Logging and evaluation
    logging_steps=10,  # Steps interval for logging
    eval_strategy="steps",  # Strategy for evaluation
    eval_steps=20,
    save_strategy="steps",  # Strategy for saving the model
    save_steps=20,
    metric_for_best_model="eval_loss",  # Metric to evaluate the best model
    greater_is_better=False,  # Whether higher metric values are better
    load_best_model_at_end=True,  # Load the best model after training
    # Mixed precision and gradient settings
    bf16=True,  # Use bfloat16 precision
    tf32=True,  # Use TensorFloat-32 precision
    max_grad_norm=0.3,  # Maximum norm for gradient clipping
    warmup_ratio=0.03,  # Ratio of total steps for warmup
    # Hub and reporting
    push_to_hub=True,  # Whether to push model to Hugging Face Hub
    report_to="wandb",  # Reporting tool for tracking metrics
    # Gradient checkpointing settings
    gradient_checkpointing_kwargs={"use_reentrant": False},  # Options for gradient checkpointing
    # Dataset configuration
    dataset_text_field="",  # Text field in dataset
    dataset_kwargs={"skip_prepare_dataset": True},  # Additional dataset options
)

training_args.remove_unused_columns = False

wandb.init(
    project="qwen2.5-vl-7b-instruct-citynav",
    name="qwen2.5-vl-7b-instruct-citynav",
    config=training_args,
)

trainer = SFTTrainer(
    model=peft_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collate_fn,
    processing_class=processor.tokenizer,
)

trainer.train()
merged_model = trainer.model.merge_and_unload()
merged_model.save_pretrained("qwen2.5-7b-instruct-citynav")
processor.save_pretrained("qwen2.5-7b-instruct-citynav")
merged_model.push_to_hub("Jise/qwen2.5-7b-instruct-citynav")
processor.push_to_hub("Jise/qwen2.5-7b-instruct-citynav")
wandb.finish()
print("Training completed and model saved.")