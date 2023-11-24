import torch
import tqdm
from gpt4all import GPT4All
from prompts import *

class RAGModel():
    def __init__(self, model_path: str) -> None:
        device = 'gpu' if torch.cuda.is_available() else 'cpu'
        self.model = GPT4All("mistral-7b-instruct-v0.1.Q4_0.gguf", model_path=model_path, device=device)
    
    def get_alternate_queries(self, question: str):
        prompt = QUERY_PROMPT.replace('$', question)
        with self.model.chat_session():
            response = self.model.generate(prompt=prompt, temp=1.0)
            queries = response.splitlines()
        
        queries.append(question)
        return queries
    
    def ask(self, 
            user: str,
            system: str = '',
            temp=0.1,
            max_tokens=500, 
            n_batch=1024
    ):
        with self.model.chat_session(system_prompt=system):
            response = self.model.generate(prompt=user, temp=temp, max_tokens=max_tokens, n_batch=n_batch)
        
        return response
    
    def enrich_samples(self, samples: list[str]):
        results = []
        for t in tqdm.tqdm(samples):
            prompt = ENRICH_PROMPT.replace('$C', t)
            with self.model.chat_session():
                response = self.model.generate(prompt=prompt, temp=1.0, max_tokens=500)
            
            inst = response.splitlines()[0].replace('Instruction: ', '').strip()
            ans = response.splitlines()[-1].replace('Response: ', '').replace('Answer: ', '').strip()
            result = ORCA_OUT.replace('$C', t).replace('$Q', inst).replace('$A', ans)
            print(result)
            results.append(result)

        return results
