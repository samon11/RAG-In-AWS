import boto3
import botocore
import json
from model import insert_document
from utils import parse_pdf, parse_txt

def split_file(event, context):
    if event['detail-type'] != 'Object Created':
        raise Exception('Invalid event type: ' + event['detail-type'])

    try:
        s3 = boto3.client('s3')

        bucket = event['detail']['bucket']['name']
        s3_object = event['detail']['object']['key']
        print('Received file to chunk: ' + s3_object)
        
        s3_response = s3.get_object(Bucket=bucket, Key=s3_object)
        s3_content = s3_response['Body'].read()
        extension = s3_object.split('.')[-1]

        if extension == 'pdf':
            chunks = parse_pdf(s3_content, max_len=3)
        else:
            chunks = parse_txt(s3_content.decode('utf-8'), max_len=3)

        print(f'Saving {len(chunks)} file chunks to SQS...')

        document_id = insert_document(bucket, s3_object, len(chunks))

        sqs = boto3.client('sqs')
        queue_url = 'QUEUE_URL'

        entries = []
        for i, chunk in enumerate(chunks):
            body = {
                'documentId': document_id,
                'filename': s3_object,
                'chunk': chunk,
                'chunkId': s3_object + '-' + str(i)
            }

            entry = {
                'Id': str(i),
                'MessageBody': json.dumps(body)
            }
            entries.append(entry)
        
        batches = [entries[i:i + 10] for i in range(0, len(entries), 10)]
        for batch in batches:
            sqs.send_message_batch(QueueUrl=queue_url, Entries=batch)

        # run task to generate embeddings from ecs fargate cluster
        ecs = boto3.client('ecs')
        response = ecs.run_task(
            cluster='CLUSTER_ARN',
            capacityProviderStrategy=[{'capacityProvider': 'FARGATE_SPOT', 'weight': 1, 'base': 0}],
            taskDefinition='TASK_NAME',
            count=1,
            platformVersion='LATEST',
            networkConfiguration={
                'awsvpcConfiguration': {
                    'subnets': [
                        'subnet-xxxxxxx',
                        'subnet-xxxxxxx'
                    ],
                    'securityGroups': [
                        'sg-xxxxxx'
                    ],
                    'assignPublicIp': 'DISABLED'
                }
            }
        )

        if 'failures' in response and len(response['failures']) > 0:
            raise Exception('Error starting task: ' + str(response['failures']))

        print('Task started: ' + str(response))
        return json.dumps({'documentId': document_id})

    except Exception as e:
        print('Error splitting file: ' + str(e))
        raise e


if __name__ == "__main__":
    a = json.loads('{"detail-type": "Object Created", "detail": {"bucket": {"name": "bucket-mjs-rag-doc-store"}, "object": {"key": "docs/redeemer.pdf"}}}')
    split_file(a, "test")