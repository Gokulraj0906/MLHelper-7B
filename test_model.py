import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import logging

# --- Configuration ---
BASE_MODEL_NAME = "gpt2-medium"
ADAPTER_PATH = "./mlai_lora_adapter"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 100

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Load Base Model ---
logger.info("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME)
base_model.to(DEVICE)

# --- Load LoRA Adapter ---
logger.info("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()
model.to(DEVICE)

# --- Load Tokenizer ---
logger.info("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# --- Text Generation Function ---
def generate_response(prompt, max_new_tokens=MAX_NEW_TOKENS):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded

# --- Test Prompts ---
test_prompts = [
    "What is machine learning?",
    "Explain neural networks.",
    "What is deep learning?",
    "What is Caffe?",
    "Difference between supervised and unsupervised learning?"
]

# --- Run Tests ---
if __name__ == "__main__":
    for prompt in test_prompts:
        logger.info(f"\nPrompt: {prompt}")
        response = generate_response(prompt)
        logger.info(f"Response:\n{response}\n{'-'*80}")
