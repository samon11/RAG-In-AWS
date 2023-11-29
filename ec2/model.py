from peewee import *
import os
import json
import boto3
from botocore.exceptions import ClientError
from pgvector.peewee import VectorField

def get_password():
    secret_name = "SECRET_NAME"
    region_name = "REGION"

    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except ClientError as e:
        raise e

    secret = json.loads(get_secret_value_response['SecretString'])
    return secret['password']

password = get_password()
host = os.getenv('RDS_HOST', 'localhost')
db = PostgresqlDatabase("doc_store", user="postgres", password=password, host=host, port="5432")

class Status:
    PENDING = 0
    COMPLETE = 1
    ERROR = 2

class BaseModel(Model):
    class Meta:
        database = db

class Document(BaseModel):
    blob_url = CharField()
    created_on = DateTimeField(constraints=[SQL("DEFAULT now()")], null=True)
    document_id = AutoField()
    chunk_count = IntegerField(null=True)
    status = IntegerField(constraints=[SQL("DEFAULT 0")])
    title = CharField()

    class Meta:
        table_name = 'document'

class DocVector(BaseModel):
    created_on = DateTimeField(constraints=[SQL("DEFAULT now()")], null=True)
    document = ForeignKeyField(column_name='document_id', field='document_id', model=Document)
    embedding = VectorField(dimensions=768)
    text = CharField()
    doc_vector_id = BigAutoField()
    chunk_id = CharField()

    class Meta:
        table_name = 'doc_vector'

def insert_doc_vector(work_item, embedding):
    # only insert if embedding doesn't exist
    existing = DocVector.select().where(DocVector.chunk_id == work_item['chunkId'])
    if existing.exists():
        return existing.get().doc_vector_id

    doc_vector = DocVector(document_id=work_item['documentId'], text=work_item['chunk'], embedding=embedding, chunk_id=work_item['chunkId'])
    doc_vector.save()
    return doc_vector.doc_vector_id

def insert_vector_batch(work_items, embeddings):
    models = [
        {
            "document_id":work_item['documentId'], 
            "text":work_item['chunk'][:2048], 
            "embedding":embedding, 
            "chunk_id":work_item['chunkId']
        }
        for work_item, embedding in zip(work_items, embeddings)]

    with db.atomic():
        return DocVector.insert_many(models).execute()

def query_embeddings(query_vector, n=5):
    # get nearest neighbors
    results = DocVector.select().join(Document).where(
        DocVector.embedding.l2_distance(query_vector) < 0.6
        ).order_by(
            DocVector.embedding.l2_distance(query_vector)
        ).limit(n)
    
    return results
