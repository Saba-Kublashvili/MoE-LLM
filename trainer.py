import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import logging
from tqdm import tqdm
import math
from config import ModelConfig
from model import SmartLLM

# Optional imports
try:
    from lion_pytorch import Lion
    LION_AVAILABLE = True
except ImportError:
    LION_AVAILABLE = False

logger = logging.getLogger(__name__)
IGNORE_INDEX = -100

class ConversationDataset(Dataset):
    def __init__(self, conversations, tokenizer, max_length=512):
        self.conversations = conversations
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.conversations)
    
    def __getitem__(self, idx):
        conv = self.conversations[idx]
        input_ids = [self.tokenizer.get_special_token_id('<bos>')]
        labels = [IGNORE_INDEX]

        for turn in conv:
            role = turn['role']
            content = turn['content']
            message = f"<{role}>{content}"
            message_ids = self.tokenizer.encode(message)
            input_ids.extend(message_ids)
            
            if role == 'assistant':
                labels.extend(message_ids)
            else:
                labels.extend([IGNORE_INDEX] * len(message_ids))

        input_ids.append(self.tokenizer.get_special_token_id('<eos>'))
        labels.append(self.tokenizer.get_special_token_id('<eos>'))

        if len(input_ids) > self.max_length:
            input_ids = input_ids[-self.max_length:]
            labels = labels[-self.max_length:]

        model_input = torch.tensor(input_ids[:-1], dtype=torch.long)
        model_target = torch.tensor(labels[1:], dtype=torch.long)
        return {'input_ids': model_input, 'targets': model_target}

class CurriculumTrainer:
    def __init__(self, model: SmartLLM, tokenizer, config: ModelConfig, device='cuda'):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        
        self.accumulation_steps = 8
        self.optimizer = self._setup_optimizer("AdamW")
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=500, T_mult=2)
        
        self.use_amp = (self.device == 'cuda')
        self.amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        
        self.step = 0
        self.curriculum_stages = [
            {"max_len": 256, "batch_size": 16, "threshold": 50.0},
            {"max_len": 512, "batch_size": 8, "threshold": 25.0},
            {"max_len": 1024, "batch_size": 4, "threshold": 15.0}
        ]

    def _setup_optimizer(self, optimizer_name: str = "AdamW"):
        if optimizer_name == "Lion" and LION_AVAILABLE:
            logger.info("Using Lion optimizer")
            return Lion(self.model.parameters(), lr=3e-5, weight_decay=0.1)
        return optim.AdamW(self.model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)

    def _collate_fn(self, batch):
        max_len = max(len(item['input_ids']) for item in batch)
        input_ids, targets = [], []
        pad_token_id = self.tokenizer.get_special_token_id('<pad>')
        
        for item in batch:
            input_pad = max_len - len(item['input_ids'])
            target_pad = max_len - len(item['targets'])
            
            input_ids.append(torch.cat([item['input_ids'], torch.full((input_pad,), pad_token_id, dtype=torch.long)]))
            targets.append(torch.cat([item['targets'], torch.full((target_pad,), IGNORE_INDEX, dtype=torch.long)]))
            
        return {'input_ids': torch.stack(input_ids), 'targets': torch.stack(targets)}

    def _train_epoch(self, dataloader, epoch, stage):
        self.model.train()
        pbar = tqdm(dataloader, desc=f"Stage {stage}, Epoch {epoch}")
        
        for i, batch in enumerate(pbar):
            with torch.autocast(device_type=self.device.split(':')[0], dtype=self.amp_dtype, enabled=self.use_amp):
                _, loss, _ = self.model(
                    batch['input_ids'].to(self.device),
                    batch['targets'].to(self.device)
                )
                loss = loss / self.accumulation_steps
            
            self.scaler.scale(loss).backward()
            
            if (i + 1) % self.accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
                self.step += 1
            
            pbar.set_postfix({'loss': loss.item() * self.accumulation_steps, 'lr': self.scheduler.get_last_lr()[0]})

    def _validate(self, val_data, max_length):
        self.model.eval()
        val_dataset = ConversationDataset(val_data, self.tokenizer, max_length)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=self._collate_fn)
        
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in val_loader:
                with torch.autocast(device_type=self.device.split(':')[0], dtype=self.amp_dtype, enabled=self.use_amp):
                    _, loss, _ = self.model(
                        batch['input_ids'].to(self.device),
                        batch['targets'].to(self.device)
                    )
                
                num_tokens = (batch['targets'] != IGNORE_INDEX).sum().item()
                if num_tokens > 0:
                    total_loss += loss.item() * num_tokens
                    total_tokens += num_tokens
        
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        return math.exp(avg_loss) if avg_loss != float('inf') else float('inf')

    def train_with_curriculum(self, train_data, val_data):
        current_stage_idx = 0
        while current_stage_idx < len(self.curriculum_stages):
            stage_config = self.curriculum_stages[current_stage_idx]
            stage_num = current_stage_idx + 1
            logger.info(f"--- Starting Curriculum Stage {stage_num} ---")
            
            dataset = ConversationDataset(train_data, self.tokenizer, stage_config["max_len"])
            dataloader = DataLoader(dataset, batch_size=stage_config["batch_size"], shuffle=True, collate_fn=self._collate_fn, num_workers=0)
            
            self._train_epoch(dataloader, epoch=1, stage=stage_num)
            perplexity = self._validate(val_data, stage_config["max_len"])
            logger.info(f"Stage {stage_num} Validation Perplexity: {perplexity:.2f}")
            
            if perplexity <= stage_config['threshold']:
                logger.info("Threshold met. Progressing.")
                current_stage_idx += 1
            else:
                logger.info("Threshold not met. Repeating stage.")

    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.use_amp else None,
            'config': self.config,
            'step': self.step,
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if self.use_amp and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.step = checkpoint.get('step', 0)
        logger.info(f"Model loaded from {path}")
        
    def get_training_stats(self):
        return {
            'step': self.step,
            'lr': self.scheduler.get_last_lr()[0],
            'device': self.device,
            'mixed_precision': self.use_amp
        }
