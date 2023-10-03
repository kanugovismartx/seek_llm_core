import traceback
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from fastapi import FastAPI
import qdrant_client
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
import os

os.environ['QDRANT_HOST'] = "https://c00712ba-0b74-414e-8c35-9f82a5aa1d19.us-east-1-0.aws.cloud.qdrant.io:6333"
os.environ['QDRANT_API_KEY'] = "-9KWMiKJPb6d1NaCGpeR6-UUdaKSTMWCXo6yAALQMFlrVb61yhJquA"
os.environ["OPENAI_API_KEY"] = "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"
os.environ['QDRANT_COLLECTION'] = 'smartx_gen_ai_core_vector_db'

OPEN_AI_KEY = os.getenv("OPEN_AI_KEY")
QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_API_KEY= os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION")


app = FastAPI()


class Item(BaseModel):
    type: str
    payload: str = None


def get_answers(query):
    client = qdrant_client.QdrantClient(url=QDRANT_HOST, api_key=QDRANT_API_KEY,)
    vectors_config = qdrant_client.http.models.VectorParams(
                                                            size=1536,
                                                            distance=qdrant_client.http.models.Distance.COSINE
                                                            )
    client.recreate_collection(collection_name=QDRANT_COLLECTION, vectors_config=vectors_config,)
    embeddings = OpenAIEmbeddings()
    vector_store = Qdrant(
                            client=client,
                            collection_name=QDRANT_COLLECTION,
                            embeddings=embeddings
    )
    qa = RetrievalQA.from_chain_type(
                                    llm=OpenAI(),
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
    response = get_answers(query)
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

    res_msg = get_answers(item.payload)
    print(res_msg)
    return JSONResponse(content=res_msg, status_code=response_code)


if __name__ == "__main__":
    print("Starting Service......")
    uvicorn.run("server:app", host="0.0.0.0", port=8080, reload=True)


