from embedder import FlagEmbedding
import boto3
from model import insert_doc_vector
import json

def main():
    model = FlagEmbedding(base_path='/models/')
    print('loaded model')

    client = boto3.client('sqs')
    queue_url = 'QUEUE_URL'

    while True:
        response = client.receive_message(QueueUrl=queue_url, MaxNumberOfMessages=10)
        messages = response.get('Messages', [])
        
        if len(messages) == 0:
            print("No messages received, exiting")
            break
    
        for message in messages:
            body = json.loads(message['Body'])
            chunk = body['chunk']
            embedding = model.embed([chunk], mode='key')[0]
            insert_doc_vector(body, embedding)
            client.delete_message(QueueUrl=queue_url, ReceiptHandle=message['ReceiptHandle'])
        
        print("Finished message batch, waiting for more")

if __name__ == "__main__":
    main()
