from embedder import FlagEmbedding
import boto3
from model import insert_vector_batch
import json
import time

def main():
    model = FlagEmbedding(base_path='/models/')
    print('loaded model')

    client = boto3.client('sqs')
    queue_url = 'QUEUE_URL'

    while True:
        response = client.receive_message(QueueUrl=queue_url, MaxNumberOfMessages=10)
        messages = response.get('Messages', [])
        
        if len(messages) == 0:
            # ensure no messages are left in queue before exiting
            queue_attributes = client.get_queue_attributes(QueueUrl=queue_url, AttributeNames=['ApproximateNumberOfMessages'])
            n_messages = int(queue_attributes['Attributes']['ApproximateNumberOfMessages'])

            if n_messages > 0:
                print(f"Queue has {n_messages} messages, waiting for more")
                time.sleep(1)
                continue
            else:
                print("No messages received, exiting")
                break

        msg_batch = [json.loads(m['Body']) for m in messages]
        chunks = [m['chunk'] for m in msg_batch]
        embeddings = model.embed(chunks, mode='key')
        try:
            insert_vector_batch(msg_batch, embeddings)
            client.delete_message_batch(QueueUrl=queue_url, Entries=[{'Id': m['MessageId'], 'ReceiptHandle': m['ReceiptHandle']} for m in messages])
        except Exception as e:
            print(f"Error processing message batch: {e}")
            continue

        print("Finished message batch, waiting for more")

if __name__ == "__main__":
    main()
