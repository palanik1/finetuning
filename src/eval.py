from tqdm import tqdm
import torch
from peft import AutoPeftModelForCausalLM,PeftModel
from transformers import AutoTokenizer,pipeline, AutoModelForCausalLM,BitsAndBytesConfig
from datasets import load_dataset
from random import randint
from trl import setup_chat_format
from tqdm import tqdm

model_id = "codellama/CodeLlama-7b-hf" # or `mistralai/Mistral-7B-v0.1`

# BitsAndBytesConfig int-4 config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

# Load model and tokenizer
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    #attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.padding_side = 'right' # to prevent warnings

# # set chat template to OAI chatML, remove if you start from a fine-tuned model
base_model, tokenizer = setup_chat_format(base_model, tokenizer)
                                                                                                                           1,12          To
peft_model_path = "code-llama-7b-text-to-sql"
#code-llama-7b-text-to-sql

model = PeftModel.from_pretrained(base_model, peft_model_path)

#model = AutoPeftModelForCausalLM(
#    peft_model_path
    #local_files_only=True,
    #load_in_4bit=True,
    #device_map = "auto",
    #torch_dtype = torch.float16
#        )


tokenizer = AutoTokenizer.from_pretrained(peft_model_path)
pipe = pipeline("text-generation", model = model, tokenizer = tokenizer)

eval_dataset = load_dataset("json", data_files = "./test_dataset.json",split = "train")
rand_idx = randint(0, len(eval_dataset))

prompt = pipe.tokenizer.apply_chat_template(eval_dataset[rand_idx]["messages"][0:2], tokenize= False, add_generation_prompt=True)

outputs = pipe(prompt, max_new_tokens=256, do_sample=False, temperature=0.1, top_k=50, top_p=0.1, eos_token_id=pipe.tokenizer.eos_token_id, pad_token_id=pipe.tokenizer.pad_token_id)

print(f"Query:\n{eval_dataset[rand_idx]['messages'][1]['content']}")
print(f"Original Answer:\n{eval_dataset[rand_idx]['messages'][2]['content']}")
print(f"Generated Answer:\n{outputs[0]['generated_text'][len(prompt):].strip()}")


def evaluate(sample):
    prompt = pipe.tokenizer.apply_chat_template(sample["messages"][:2], tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95, eos_token_id=pipe.tokenizer.eos_token_id, pad_token_id=pipe.tokenizer.pad_token_id)
    predicted_answer = outputs[0]['generated_text'][len(prompt):].strip()
    if predicted_answer == sample["messages"][2]["content"]:
        return 1
    else:
        return 0

success_rate = []
num_eval_samples = 1000
for sample in tqdm(eval_dataset.shuffle().select(range(num_eval_samples))):
    success_rate.append(evaluate(sample))
accuracy = sum(success_rate/len(success_rate)
               )
print(f"Accuracy: {accuracy}")
