import torch
import tqdm
from transformers import AutoTokenizer, AutoModel
from prompts import INSTRUCTIONS

class FlagEmbedding():
    def __init__(self, n_batch=32, base_path = '') -> None:
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(base_path + 'BAAI/llm-embedder')
        self.model = AutoModel.from_pretrained(base_path + 'BAAI/llm-embedder').to(self.device)
        self.prefix = INSTRUCTIONS['qa']
        self.batch_size = n_batch

    def embed(self, texts, mode: str = 'key'):
        if mode == 'key':
            text_inputs = [self.prefix["key"] + text for text in texts]
        elif mode == 'query':
            text_inputs = [self.prefix["query"] + text for text in texts]
        else:
            raise NotImplementedError(f"mode {mode} not implemented")

        inputs = self.tokenizer(text_inputs, padding=True, return_tensors='pt', max_length=512, truncation=True).to(self.device)
        embeddings = []
        for i in tqdm.trange(0, len(inputs['input_ids']), self.batch_size):
            batch_inputs = {k: v[i:i+self.batch_size] for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**batch_inputs)
                batch = outputs.last_hidden_state[:, 0]
                batch = torch.nn.functional.normalize(batch, p=2, dim=1)
                embeddings.extend(batch.tolist())

        return embeddings
