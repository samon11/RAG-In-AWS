from flask import Flask, request, jsonify
from model import query_embeddings
from embedder import FlagEmbedding
from llm import RAGModel
import numpy as np
from prompts import FS_USER, FS_SYSTEM

app = Flask(__name__)
embedder = FlagEmbedding(base_path='/models/')
llm = RAGModel(model_path='/models')

def get_context(query, n=5):
    llm_queries = llm.get_alternate_queries(query)

    vectors = embedder.embed(llm_queries, mode='query')

    # RAG fusion
    results = {'documents': [], 'distances': []}
    for v in vectors:
        matches = query_embeddings(v, n=n)
        results['documents'].append([r.text for r in matches])
        results['distances'].append([np.linalg.norm(r.embedding - v) for r in matches])

        if len(results['distances'][-1]) > 0:
            print('Smallest distance: ', min(results['distances'][-1]))

    fused_scores = {}
    for idx in range(len(llm_queries)):
        for j, doc in enumerate(results['documents'][idx]):
            if doc not in fused_scores:
                fused_scores[doc] = 0
            rank = results['distances'][idx][j]
            fused_scores[doc] += 1 / (rank + 60)

    reranked = sorted(fused_scores.items(), key=lambda x:x[1])
    context = [{'doc': c[0].strip(), 'score': c[1]} for c in reranked[:n]]
    return context

@app.route('/api/search', methods=['POST'])
def search():
    data = request.get_json()
    context = get_context(data['query'], n=5)
    if len(context) == 0:
        return jsonify({'error': 'No relevant docs found'}), 404

    return jsonify(context)

@app.route('/api/ask', methods=['POST'])
def ask():
    data = request.get_json()
    question = data['question']
    context = get_context(question, n=5)
    if len(context) == 0:
        return jsonify({'error': 'No relevant docs found'}), 404

    context_docs = '\n'.join([c['doc'] for c in context])
    prompt = FS_USER.replace('$C', context_docs).replace('$Q', question)
    response = llm.ask(prompt, FS_SYSTEM, temp=data.get('temp', 0.5), max_tokens=data.get('max_tokens', 500))
    return jsonify({'response': response})

@app.route('/hc', methods=['GET'])
def hc():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
