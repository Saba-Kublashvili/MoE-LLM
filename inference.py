import torch
import torch.nn.functional as F
import logging
from tokenization import SmartBPETokenizer

logger = logging.getLogger(__name__)

class SmartChatBot:
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
        
        self.conversation_history = []
        self.max_history_length = 2048
        
        self.default_params = {
            'max_length': 200,
            'temperature': 0.7,
            'top_p': 0.9,
            'top_k': 50,
            'repetition_penalty': 1.1
        }
        
        # Determine EOS tokens based on tokenizer type
        self.stop_tokens = {self.tokenizer.get_special_token_id('<eos>')}
        if isinstance(self.tokenizer, SmartBPETokenizer):
             self.stop_tokens.add(self.tokenizer.get_special_token_id('<user>'))

    def _apply_repetition_penalty(self, logits, input_ids, penalty):
        if penalty == 1.0: return logits
        score = torch.gather(logits, 1, input_ids)
        score = torch.where(score < 0, score * penalty, score / penalty)
        logits.scatter_(1, input_ids, score)
        return logits

    def _apply_top_k_filtering(self, logits, top_k):
        if top_k <= 0: return logits
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')
        return logits

    def _apply_top_p_filtering(self, logits, top_p):
        if top_p >= 1.0: return logits
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = float('-inf')
        return logits

    def _build_context(self, user_input: str) -> str:
        current_turn = f"<user>{user_input}<assistant>"
        if self.conversation_history:
            context = "".join(self.conversation_history) + current_turn
        else:
            context = current_turn
        
        context_tokens = self.tokenizer.encode(context)
        while len(context_tokens) > self.max_history_length and self.conversation_history:
            self.conversation_history.pop(0)
            context = "".join(self.conversation_history) + current_turn
            context_tokens = self.tokenizer.encode(context)
        
        return context

    def generate_response(self, user_input: str, **kwargs):
        params = {**self.default_params, **kwargs}
        context = self._build_context(user_input)
        
        start_token = '<bos>'
        if hasattr(self.tokenizer, 'special_tokens') and isinstance(self.tokenizer.special_tokens, dict):
             # Fallback tokenizer uses dict
             bos_id = self.tokenizer.special_tokens['<bos>']
        else:
             bos_id = self.tokenizer.get_special_token_id('<bos>')

        tokens = [bos_id] + self.tokenizer.encode(context)
        input_ids = torch.tensor([tokens], dtype=torch.long).to(self.device)
        
        kv_caches = None
        generated_tokens = []
        
        with torch.no_grad():
            for step in range(params['max_length']):
                logits, _, kv_caches = self.model(input_ids, kv_caches=kv_caches)
                next_token_logits = logits[:, -1, :] / params['temperature']
                
                if step > 0:
                    next_token_logits = self._apply_repetition_penalty(
                        next_token_logits, input_ids, params['repetition_penalty']
                    )
                
                next_token_logits = self._apply_top_k_filtering(next_token_logits, params['top_k'])
                next_token_logits = self._apply_top_p_filtering(next_token_logits, params['top_p'])
                
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                if next_token.item() in self.stop_tokens:
                    break
                
                generated_tokens.append(next_token.item())
                input_ids = next_token
        
        if generated_tokens:
            response = self.tokenizer.decode(generated_tokens)
            response = response.split('<user>')[0].split('<eos>')[0].strip()
        else:
            response = "..."
        
        self.conversation_history.append(f"<user>{user_input}<assistant>{response}")
        return response

    def clear_history(self):
        self.conversation_history = []
    
    def set_default_params(self, **params):
        for key, value in params.items():
            if key in self.default_params:
                self.default_params[key] = value
