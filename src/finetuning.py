from datasets import load_dataset
import transformers
import torch

from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import setup_chat_format,SFTTrainer
from transformers import TrainingArguments
from peft import LoraConfig

login(
  token="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", # ADD YOUR TOKEN HERE
  add_to_git_credential=True
)
 


system_message = """You are an text to SQL query translator. Users will ask you questions in English and you will generate a SQL query based on the provided SCHEMA.
SCHEMA:
{schema}"""


def create_conversation(sample):
    return {
    "messages": [
      {"role": "system", "content": system_message.format(schema=sample["context"])},
      {"role": "user", "content": sample["question"]},
      {"role": "assistant", "content": sample["answer"]}
    ]
    }

dataset = load_dataset("b-mc2/sql-create-context")
dataset = dataset["train"].shuffle().select(range(12500))

dataset = dataset.map(create_conversation,remove_columns=dataset.features,batched=False)

dataset = dataset.train_test_split(test_size=2500/12500)

print(f"{dataset}")


 
# save datasets to disk
dataset["train"].to_json("train_dataset.json", orient="records")
dataset["test"].to_json("test_dataset.json", orient="records")


dataset = load_dataset("json", data_files="train_dataset.json", split="train")

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16
)

model_id = "codellama/CodeLlama-7b-hf"
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map = "auto",
    torch_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(
          model_id
    )
tokenizer.padding_side = "right"
base_model,tokenizer = setup_chat_format(base_model,tokenizer)

peft_config = LoraConfig(r = 256, 
                         lora_alpha = 512,
                         lora_dropout = 0.5,
                         task_type = "CASUAL_LM",
                         target_modules = "all-linear",
                         bias = "none"
                         )

    
    
training_args = TrainingArguments(
    output_dir = "code-llama-7b-text-to-sql",
    num_train_epochs = 3,
    optim = "adamw_torch_fused",
    logging_steps = 10,
    per_device_train_batch_size = 3,
    gradient_accumulation_steps = 2,
    gradient_checkpointing = True,
    learning_rate = 2e-4,
    save_strategy = "epoch",
    bf16 = True,
    tf32 = True,
    max_grad_norm = 0.3,
    warmup_ratio = 0.03,
    lr_scheduler_type = "constant",
    push_to_hub = False,
    report_to = "tensorboard",
    remove_unused_columns=False
  )                     

trainer = SFTTrainer(
    model = base_model,
    peft_config = peft_config,
    args = training_args,
    train_dataset = dataset,
    max_seq_length = 3072,
    packing = True,
    tokenizer = tokenizer,
    dataset_kwargs={
        "add_special_tokens": False,  # We template with special tokens
        "append_concat_token": False, # No need to add additional separator token
    }

)

trainer.train()
trainer.save_model()
