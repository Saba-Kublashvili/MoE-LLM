import os
import torch
import random
import logging
import glob
from config import ModelConfig
from model import SmartLLM
from tokenization import SmartBPETokenizer, FallbackCharTokenizer, TOKENIZERS_AVAILABLE
from trainer import CurriculumTrainer
from inference import SmartChatBot

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check for datasets library
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

def format_dolly_dataset(dataset):
    formatted_data = []
    for item in dataset:
        instruction, context, response = item['instruction'], item['context'], item['response']
        user_content = f"{context}\n\n{instruction}" if context else instruction
        formatted_data.append([
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": response}
        ])
    return formatted_data

def prepare_tokenizer_training_data(dataset):
    texts = []
    for item in dataset:
        instruction, context, response = item['instruction'], item['context'], item['response']
        user_content = f"{context}\n\n{instruction}" if context else instruction
        texts.append(f"<user>{user_content}<assistant>{response}")
    return texts

def main():
    print(" Initializing Modular Smart LLM...")
    
    # 1. Configuration
    # Using the "Large" configuration described in the original requirements
    config = ModelConfig(
        vocab_size=16000 if TOKENIZERS_AVAILABLE else 32000,
        d_model=512,
        n_heads=8,
        n_kv_heads=2,
        n_layers=8,
        d_ff=1024,
        max_seq_len=1024,
        dropout=0.1,
        use_mixture_of_experts=True,
        num_experts=4,
        num_experts_per_token=2
    )

    # 2. Tokenizer Initialization
    if TOKENIZERS_AVAILABLE:
        tokenizer = SmartBPETokenizer(config.vocab_size)
        tokenizer_path = "smart-bpe-tokenizer.json"
        print("✅ Using BPE tokenizer")
    else:
        print(" Tokenizers library not found. Using fallback.")
        tokenizer = FallbackCharTokenizer(config.vocab_size)
        tokenizer_path = "fallback-tokenizer.json"

    # 3. Model & Trainer Initialization
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SmartLLM(config)
    trainer = CurriculumTrainer(model, tokenizer, config, device=device)
    
    model_path = "smart_llm_model.pth"

    # 4. Load or Train
    if os.path.exists(model_path):
        print("\n📂 Loading existing model...")
        try:
            trainer.load_model(model_path)
            if os.path.exists(tokenizer_path):
                tokenizer.load(tokenizer_path)
                print("✅ Tokenizer loaded.")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            print("🔄 Load failed. Will attempt to train.")
    
    # Logic to trigger training if model doesn't exist or failed load
    if not os.path.exists(model_path):
        print("\n🎯 Training new model...")
        if not DATASETS_AVAILABLE:
            print("❌ 'datasets' library missing. Cannot load training data.")
        else:
            try:
                # Robust dataset loading: Try local path, then HF Hub
                local_search = glob.glob(os.path.expanduser("~/**/databricks-dolly-15k.jsonl"), recursive=True)
                if local_search:
                    data_source = local_search[0]
                    logger.info(f"Loading local dataset: {data_source}")
                    dolly_dataset = load_dataset("json", data_files=data_source, split="train")
                else:
                    logger.info("Local file not found. Downloading from HuggingFace...")
                    dolly_dataset = load_dataset("databricks/dolly-15k", split="train")

                # Train Tokenizer if needed
                if TOKENIZERS_AVAILABLE and not os.path.exists(tokenizer_path):
                    logger.info("Training BPE tokenizer...")
                    texts = prepare_tokenizer_training_data(dolly_dataset)
                    tokenizer.train_from_texts(texts, tokenizer_path)
                elif os.path.exists(tokenizer_path):
                    tokenizer.load(tokenizer_path)

                # Prepare Data
                training_data = format_dolly_dataset(dolly_dataset)
                random.shuffle(training_data)
                split_idx = int(len(training_data) * 0.95)
                train_data, val_data = training_data[:split_idx], training_data[split_idx:]
                
                # Train
                trainer.train_with_curriculum(train_data, val_data)
                trainer.save_model(model_path)
                
            except Exception as e:
                logger.error(f"Training interrupted: {e}")
                print("⚠️ Proceeding with untrained model for demonstration.")

    # 5. Chat Interface
    print("\n🤖 Smart LLM Interactive Chat")
    print("Commands: 'quit', 'stats', 'clear', 'params <k>=<v>'")
    
    chatbot = SmartChatBot(model, tokenizer, device)
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            if not user_input: continue
            
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'stats':
                print(trainer.get_training_stats())
                continue
            elif user_input.lower() == 'clear':
                chatbot.clear_history()
                print("History cleared.")
                continue
            elif user_input.lower().startswith('params '):
                try:
                    k, v = user_input[7:].split('=')
                    val = float(v) if '.' in v else int(v)
                    chatbot.set_default_params(**{k.strip(): val})
                    print(f"Set {k} to {val}")
                except:
                    print("Invalid format. Use: params temperature=0.8")
                continue

            response = chatbot.generate_response(user_input)
            print(f"🤖 Assistant: {response}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    # Environment optimizations
    os.environ["OMP_NUM_THREADS"] = "8"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.manual_seed(42)
    random.seed(42)
    
    main()
