# import torch
# import logging
# from transformers import (
#     AutoTokenizer, 
#     AutoModelForCausalLM,
#     DataCollatorForLanguageModeling,
#     TrainingArguments,
#     Trainer,
#     BitsAndBytesConfig,
#     pipeline
# )
# from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
# from datasets import load_dataset, Dataset
# import json
# import os

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# # SOLUTION 1: Use truly open-source alternatives (no gating)
# ALTERNATIVE_MODELS = {
#     "gpt2": "gpt2-medium",
#     "distilgpt2": "distilgpt2",
#     "bloom": "bigscience/bloom-560m",
#     "opt": "facebook/opt-350m",
#     "pythia": "EleutherAI/pythia-410m",
#     "gpt_neo": "EleutherAI/gpt-neo-1.3B",
#     "t5": "t5-small",
#     "flan_t5": "google/flan-t5-small",
# }

# # Select model
# MODEL_NAME = ALTERNATIVE_MODELS["gpt2"]

# # Constants
# DATASET_NAME = "squad"
# DOMAIN_DOCS_PATH = "domain_docs.jsonl"
# MAX_LENGTH = 512  # Reduced for better memory management
# LORA_R = 16       # Reduced for stability
# LORA_ALPHA = 32   # Reduced proportionally
# LORA_DROPOUT = 0.1

# # --- Helper Functions ---
# def create_sample_domain_docs():
#     """Create sample ML/AI domain documents if file doesn't exist"""
#     if not os.path.exists(DOMAIN_DOCS_PATH):
#         sample_docs = [
#             {
#                 "text": "Machine Learning is a subset of artificial intelligence that focuses on algorithms that can learn from data. Common algorithms include linear regression, decision trees, random forests, and neural networks.",
#                 "title": "ML Basics"
#             },
#             {
#                 "text": "Deep Learning uses neural networks with multiple layers to learn complex patterns. Key architectures include CNNs for image processing, RNNs for sequential data, and Transformers for language tasks.",
#                 "title": "Deep Learning"
#             },
#             {
#                 "text": "Gradient Boosting builds models sequentially, with each new model correcting errors from previous ones. Random Forests use multiple decision trees trained on different subsets of data and average their predictions.",
#                 "title": "Ensemble Methods"
#             },
#             {
#                 "text": "Natural Language Processing involves techniques for understanding and generating human language. Key tasks include tokenization, named entity recognition, sentiment analysis, and text generation.",
#                 "title": "NLP"
#             }
#         ]
        
#         with open(DOMAIN_DOCS_PATH, 'w') as f:
#             for doc in sample_docs:
#                 f.write(json.dumps(doc) + '\n')
#         logger.info(f"Created sample domain documents at {DOMAIN_DOCS_PATH}")

# def setup_huggingface_auth():
#     """Setup Hugging Face authentication if token is available"""
#     hf_token = os.getenv('HF_TOKEN')
#     if hf_token:
#         try:
#             from huggingface_hub import login
#             login(token=hf_token)
#             logger.info("Logged in to Hugging Face")
#             return True
#         except Exception as e:
#             logger.warning(f"HF login failed: {e}")
#     return False

# # --- 1. Load Model with QLoRA ---
# logger.info(f"Loading model: {MODEL_NAME}")
# try:
#     setup_huggingface_auth()
    
#     # Only use quantization if CUDA is available
#     if torch.cuda.is_available():
#         bnb_config = BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_compute_dtype=torch.float16,
#             bnb_4bit_use_double_quant=True,
#             bnb_4bit_quant_type="nf4"
#         )
#         quantization_config = bnb_config
#         torch_dtype = torch.float16
#     else:
#         quantization_config = None
#         torch_dtype = torch.float32
    
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
#     # Handle tokenizer padding token
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
#     tokenizer.padding_side = "right"
    
#     model = AutoModelForCausalLM.from_pretrained(
#         MODEL_NAME,
#         quantization_config=quantization_config,
#         device_map="auto" if torch.cuda.is_available() else None,
#         torch_dtype=torch_dtype,
#         trust_remote_code=True
#     )
    
#     if torch.cuda.is_available() and quantization_config:
#         model = prepare_model_for_kbit_training(model)
    
#     logger.info("Model loaded successfully!")
    
# except Exception as e:
#     logger.error(f"Failed to load {MODEL_NAME}. Error: {str(e)}")
#     logger.info("Trying fallback model...")
    
#     # Fallback to GPT-2
#     MODEL_NAME = "gpt2"
#     try:
#         tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
#         if tokenizer.pad_token is None:
#             tokenizer.pad_token = tokenizer.eos_token
#         tokenizer.padding_side = "right"
        
#         model = AutoModelForCausalLM.from_pretrained(
#             MODEL_NAME,
#             torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
#             device_map="auto" if torch.cuda.is_available() else None
#         )
#         logger.info("Fallback model loaded successfully!")
#     except Exception as e2:
#         logger.exception("All model loading attempts failed")
#         raise

# # --- 2. Configure LoRA ---
# logger.info("Configuring LoRA for domain adaptation")
# use_lora = False
# try:
#     # Define target modules based on model type
#     model_type = getattr(model.config, 'model_type', 'unknown')
#     logger.info(f"Model type: {model_type}")
    
#     if model_type == 'gpt2':
#         target_modules = ["c_attn", "c_proj"]
#     elif model_type == 'bloom':
#         target_modules = ["query_key_value", "dense"]
#     elif model_type == 'opt':
#         target_modules = ["q_proj", "k_proj", "v_proj", "out_proj"]
#     elif model_type == 't5':
#         target_modules = ["q", "k", "v", "o"]
#     else:
#         # Find attention modules dynamically
#         target_modules = []
#         for name, module in model.named_modules():
#             if any(target in name.lower() for target in ["attn", "attention"]):
#                 module_name = name.split('.')[-1]
#                 if module_name not in target_modules:
#                     target_modules.append(module_name)
#         target_modules = target_modules[:2]  # Limit to 2 modules
    
#     if target_modules:
#         logger.info(f"Target modules for LoRA: {target_modules}")
        
#         lora_config = LoraConfig(
#             r=LORA_R,
#             lora_alpha=LORA_ALPHA,
#             target_modules=target_modules,
#             lora_dropout=LORA_DROPOUT,
#             bias="none",
#             task_type="CAUSAL_LM",
#             inference_mode=False
#         )
#         model = get_peft_model(model, lora_config)
#         model.print_trainable_parameters()
#         use_lora = True
#         logger.info("LoRA configuration successful!")
#     else:
#         logger.warning("No suitable target modules found. Continuing without LoRA.")
        
# except Exception as e:
#     logger.warning(f"LoRA configuration failed: {e}. Continuing without LoRA...")

# # --- 3. Domain-Specific Dataset Preparation ---
# logger.info("Preparing domain dataset")
# try:
#     create_sample_domain_docs()
    
#     # Load dataset
#     dataset = load_dataset("squad", split="train[:500]")  # Small subset
    
#     def preprocess_function(examples):
#         inputs = []
#         for question, context, answer in zip(examples["question"], examples["context"], examples["answers"]):
#             answer_text = answer['text'][0] if answer['text'] else 'No answer'
#             formatted_input = f"Question: {question}\nContext: {context}\nAnswer: {answer_text}"
#             inputs.append(formatted_input)
        
#         tokenized = tokenizer(
#             inputs,
#             max_length=MAX_LENGTH,
#             truncation=True,
#             padding="max_length",
#             return_tensors=None  # Don't return tensors here
#         )
        
#         # Create labels
#         tokenized["labels"] = tokenized["input_ids"].copy()
#         return tokenized

#     tokenized_dataset = dataset.map(
#         preprocess_function,
#         batched=True,
#         remove_columns=dataset.column_names
#     ).train_test_split(test_size=0.1)
    
#     logger.info(f"Dataset prepared with {len(tokenized_dataset['train'])} training samples")
    
# except Exception as e:
#     logger.exception("Dataset processing failed")
#     raise

# # --- 4. Fine-tuning Configuration ---
# logger.info("Configuring training")
# data_collator = DataCollatorForLanguageModeling(
#     tokenizer=tokenizer,
#     mlm=False
# )

# # FIXED: Use correct parameter names for your transformers version
# training_args = TrainingArguments(
#     output_dir="./mlai_finetuned",
#     num_train_epochs=1,
#     per_device_train_batch_size=1,
#     gradient_accumulation_steps=4,
#     learning_rate=5e-5,
#     fp16=torch.cuda.is_available(),
#     logging_steps=10,
#     # FIXED: Use 'eval_strategy' instead of 'evaluation_strategy'
#     eval_strategy="steps",  # Changed from evaluation_strategy
#     eval_steps=50,
#     save_strategy="steps",
#     save_steps=50,  # Match eval_steps
#     load_best_model_at_end=True,
#     metric_for_best_model="eval_loss",
#     greater_is_better=False,
#     warmup_ratio=0.05,
#     max_steps=100,
#     report_to=None,  # Disable wandb/tensorboard
#     dataloader_pin_memory=False,  # Reduce memory usage
#     remove_unused_columns=False,
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_dataset["train"],
#     eval_dataset=tokenized_dataset["test"],
#     data_collator=data_collator,
#     tokenizer=tokenizer
# )

# # --- 5. Simple RAG System ---
# class SimpleRAGGenerator:
#     def __init__(self, model, tokenizer, domain_docs_path):
#         self.model = model
#         self.tokenizer = tokenizer
#         self.domain_docs = self.load_domain_docs(domain_docs_path)
        
#         # Create pipeline with error handling
#         try:
#             self.pipeline = pipeline(
#                 "text-generation",
#                 model=model,
#                 tokenizer=tokenizer,
#                 device=0 if torch.cuda.is_available() else -1,
#                 torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
#             )
#         except Exception as e:
#             logger.warning(f"Pipeline creation failed: {e}. Using basic generation.")
#             self.pipeline = None
    
#     def load_domain_docs(self, docs_path):
#         """Load domain documents from JSONL file"""
#         docs = []
#         try:
#             with open(docs_path, 'r') as f:
#                 for line in f:
#                     docs.append(json.loads(line.strip()))
#         except FileNotFoundError:
#             logger.warning(f"Domain docs file {docs_path} not found.")
#         return docs
    
#     def simple_retrieve(self, query, top_k=2):
#         """Simple keyword-based retrieval"""
#         query_words = set(query.lower().split())
#         scored_docs = []
        
#         for doc in self.domain_docs:
#             doc_words = set(doc["text"].lower().split())
#             overlap = len(query_words.intersection(doc_words))
#             if overlap > 0:
#                 scored_docs.append((overlap, doc))
        
#         scored_docs.sort(reverse=True, key=lambda x: x[0])
#         return [doc for _, doc in scored_docs[:top_k]]
    
#     def generate(self, query: str, max_length: int = 50) -> str:
#         """Generate response using retrieved context"""
#         retrieved_docs = self.simple_retrieve(query)
        
#         context = "\n".join([doc["text"] for doc in retrieved_docs])
        
#         if context:
#             prompt = f"Context: {context[:200]}...\n\nQuestion: {query}\nAnswer:"
#         else:
#             prompt = f"Question: {query}\nAnswer:"
        
#         try:
#             if self.pipeline:
#                 output = self.pipeline(
#                     prompt,
#                     max_new_tokens=max_length,
#                     temperature=0.7,
#                     do_sample=True,
#                     return_full_text=False,
#                     pad_token_id=self.tokenizer.eos_token_id
#                 )
#                 return output[0]['generated_text'].strip()
#             else:
#                 # Fallback generation
#                 inputs = self.tokenizer(prompt, return_tensors="pt", max_length=MAX_LENGTH, truncation=True)
#                 if torch.cuda.is_available():
#                     inputs = inputs.to('cuda')
                
#                 with torch.no_grad():
#                     outputs = self.model.generate(
#                         **inputs,
#                         max_new_tokens=max_length,
#                         temperature=0.7,
#                         do_sample=True,
#                         pad_token_id=self.tokenizer.eos_token_id,
#                         eos_token_id=self.tokenizer.eos_token_id
#                     )
                
#                 response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
#                 response = response[len(prompt):].strip()
#                 return response
                
#         except Exception as e:
#             logger.error(f"Generation failed: {e}")
#             return f"Error generating response: {str(e)}"

# # --- 6. Test the system ---
# logger.info("Testing the system")
# try:
#     rag_generator = SimpleRAGGenerator(model, tokenizer, DOMAIN_DOCS_PATH)
    
#     test_queries = [
#         "What is machine learning?",
#         "Explain neural networks",
#         "What is deep learning?"
#     ]
    
#     for query in test_queries:
#         logger.info(f"\nQUERY: {query}")
#         response = rag_generator.generate(query)
#         logger.info(f"RESPONSE: {response}")
        
# except Exception as e:
#     logger.exception("Testing failed")

# # --- 7. Training (optional) ---
# def run_training():
#     """Run the training process"""
#     try:
#         logger.info("Starting training...")
#         trainer.train()
        
#         # Save the model
#         output_dir = "./mlai_lora_adapter" if use_lora else "./mlai_finetuned_model"
#         model.save_pretrained(output_dir)
#         tokenizer.save_pretrained(output_dir)
#         logger.info(f"Model saved to {output_dir}")
        
#     except Exception as e:
#         logger.exception("Training failed")

# # Uncomment the line below to start training
# run_training()

# logger.info("Setup complete!")

# print("""
# SETUP COMPLETE!

# Key fixes applied:
# 1. Fixed training arguments - evaluation_strategy and save_strategy now match
# 2. Reduced model complexity for better stability
# 3. Added proper error handling throughout
# 4. Memory optimizations for both CPU and GPU
# 5. Fixed tensor handling issues
# 6. Improved LoRA configuration

# To start training, uncomment the last line: run_training()

# Test the RAG system with the generated responses above.
# """)

import os
import json
import logging
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# === Configuration ===
MODEL_NAME = "gpt2-medium"
DOC_PATH = "main_doc.jsonl"
OUTPUT_DIR = "./mlhelper7b_lora"
MAX_LENGTH = 512
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
USE_CUDA = torch.cuda.is_available()

# === Logging Setup ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# === Load Tokenizer and Model ===
logger.info("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
) if USE_CUDA else None

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config if USE_CUDA else None,
    device_map="auto" if USE_CUDA else None,
    torch_dtype=torch.float16 if USE_CUDA else torch.float32
)

if USE_CUDA:
    model = prepare_model_for_kbit_training(model)

# === Apply LoRA ===
logger.info("Applying LoRA...")
target_modules = ["c_attn", "c_proj"] if "gpt2" in MODEL_NAME else ["q_proj", "k_proj", "v_proj", "out_proj"]
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=target_modules,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
    inference_mode=False
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# === Load and Prepare Dataset ===
logger.info(f"Loading domain dataset from {DOC_PATH}...")
with open(DOC_PATH, "r", encoding="utf-8") as f:
    samples = [json.loads(line.strip()) for line in f]

dataset = Dataset.from_list(samples)

def preprocess(example):
    prompt = example.get("text", "").strip()
    encoded = tokenizer(
        prompt,
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length"
    )
    encoded["labels"] = encoded["input_ids"].copy()
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "labels": encoded["labels"]
    }

logger.info("Tokenizing dataset...")
tokenized_dataset = dataset.map(preprocess, remove_columns=dataset.column_names)

# === Training Setup ===
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    fp16=USE_CUDA,
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    report_to="none",
    dataloader_pin_memory=False,
    remove_unused_columns=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# === Train ===
logger.info("Starting training...")
trainer.train()

# === Save ===
logger.info("Saving model...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
logger.info(f"Model saved at {OUTPUT_DIR}")