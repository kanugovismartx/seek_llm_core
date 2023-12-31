from fastapi.responses import JSONResponse
from pydantic import BaseModel
import urllib.parse
import uvicorn
from fastapi import FastAPI
import qdrant_client
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
import boto3
from botocore.exceptions import ClientError
import json
import os


AWS_KEY = os.environ.get("aws_key")
AWS_SECRET = os.environ.get("aws_secret")
AWS_REGION = "us-east-1"


def get_secrets():
    secret_name = "smartx_llm_core"
    region_name = "us-east-1"

    # Create a Secrets Manager client
    session = boto3.session.Session(aws_access_key_id=AWS_KEY, aws_secret_access_key=AWS_SECRET,region_name=AWS_REGION)
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )

    except ClientError as e:
        # For a list of exceptions thrown, see
        # https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
        raise e

    # Decrypts secret using the associated KMS key.
    return get_secret_value_response['SecretString']


secrets = json.loads(get_secrets())
OPEN_AI_KEY = secrets["OPENAI_API_KEY"]
QDRANT_HOST = secrets["QDRANT_HOST"]
QDRANT_API_KEY= secrets["QDRANT_API_KEY"]
QDRANT_COLLECTION = 'smartx_gen_ai_core_vector_db'


app = FastAPI()


class Item(BaseModel):
    type: str
    payload: str = None
    collection: str = None


def get_answers(query, collection):
    QDRANT_COLLECTION = collection
    print("Collection being used : "+QDRANT_COLLECTION)
    client = qdrant_client.QdrantClient(url=QDRANT_HOST, api_key=QDRANT_API_KEY,)
    vectors_config = qdrant_client.http.models.VectorParams(
                                                            size=1536,
                                                            distance=qdrant_client.http.models.Distance.COSINE
                                                            )
    #client.recreate_collection(collection_name=QDRANT_COLLECTION, vectors_config=vectors_config,)
    embeddings = OpenAIEmbeddings(openai_api_key=OPEN_AI_KEY)
    vector_store = Qdrant(
                            client=client,
                            collection_name=QDRANT_COLLECTION,
                            embeddings=embeddings
    )
    qa = RetrievalQA.from_chain_type(
                                    llm=OpenAI(openai_api_key=OPEN_AI_KEY),
                                    chain_type="stuff",
                                    retriever=vector_store.as_retriever()
                                    )

    response = qa.run(query)
    print (response)
    return response


@app.get("/")
async def index():
    return {"message": "Hello World"}


@app.get("/query/{query}")
async def query_text(query):
    original_query = urllib.parse.unquote_plus(query)
    response = get_answers(original_query)
    return {"message": "text received " + query + ", and the " + response}


@app.post("/seek/")
async def train_item(item: Item):
    # Perform some operations with the received item data
    response_code = 200

    if item.type != "query" and item.type != "chat":
        response_code = 400
        message = "type can only be either query or chat"
        response_body = {"message": message}
        return JSONResponse(content=response_body, status_code=response_code)

    res_msg = get_answers(item.payload, item.collection)
    print(res_msg)
    return JSONResponse(content=res_msg, status_code=response_code)


if __name__ == "__main__":
    print("Starting Service......")
    uvicorn.run("server:app", host="0.0.0.0", port=8080, reload=True)
