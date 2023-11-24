QUERY_PROMPT = """You are a helpful assistant that generates multiple search queries based on a single input query. Generate multiple search queries related to $. OUTPUT (5 queries):"""

FS_SYSTEM = """
You are a helpful assistant that answers questions about documents in a local file system. 
The doc snippets included may or may not be relevant to the question. You must tell the user if no relevant documents are found.
Try to ignore project asset code snippets and focus on the code snippets that are relevant to the question.
"""

FS_USER = """
Use the following docs as reference:
[DOCS]
$C
[/DOCS]
Answer the following question: $Q
"""

INSTRUCTIONS = {
    "qa": {
        "query": "Represent this query for retrieving relevant documents: ",
        "key": "Represent this document for retrieval: ",
    },
    "icl": {
        "query": "Convert this example into vector to look for useful examples: ",
        "key": "Convert this example into vector for retrieval: ",
    },
    "chat": {
        "query": "Embed this dialogue to find useful historical dialogues: ",
        "key": "Embed this historical dialogue for retrieval: ",
    },
    "lrlm": {
        "query": "Embed this text chunk for finding useful historical chunks: ",
        "key": "Embed this historical text chunk for retrieval: ",
    },
    "tool": {
        "query": "Transform this user request for fetching helpful tool descriptions: ",
        "key": "Transform this tool description for retrieval: "
    },
    "convsearch": {
        "query": "Encode this query and context for searching relevant passages: ",
        "key": "Encode this passage for retrieval: ",
    },
}
