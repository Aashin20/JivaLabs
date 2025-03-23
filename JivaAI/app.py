from flask import Flask, request, Response
from flask_cors import CORS
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import LlamaCpp

model_name = "sentence-transformers/all-MiniLM-l6-v2"
embeddings = HuggingFaceEmbeddings()
db3 = Chroma(persist_directory="./chroma_db3", embedding_function=embeddings)

def truncate_string(input_string):
    if len(input_string) > 4000:
        return "..." + input_string[-3997:]
    else:
        return input_string


callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
n_gpu_layers = 1
n_batch = 128
llm = LlamaCpp(
    model_path="openhermes-2.5-mistral-7b-16k.Q4_K_M.gguf",
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    f16_kv=True,
    temperature=0.70,
    max_tokens=800,
    n_ctx=3000,
)

app = Flask(__name__)
CORS(app)

@app.route("/", methods=['POST'])
def receive_data():
    memory = request.form.get("memory")
    truncated_mem = truncate_string(memory)
    data = request.form.get('name')
    docs = db3.similarity_search(data, k=6)
    query_res = docs[1::2]

    prompt = f"""Context: You are Dr.insights, a friendly AI medical advisor and health assistant that answers the user's health related queries
in a direct and straightforward way using the DB_examples(Dont talk about these examples in the answers though) that has some previous example 
question and answer pairs to help you respond to the user. Sometimes you may get 'Hi' or such general query from the user in which case, you will ignore
the db examples and just greet them normally. Only use the db if the user_query is relevant otherwise ignore it. Mention the URL's at the end as "references"
Give a very long, point-wise and detailed answers to the user's query.

Chat History so far:{truncated_mem}
User's query:{data}
Example Question/Answers:
(WARNING: DO NOT USE THESE DB EXAMPLES IF THE USER_QUERY IS NOT RELEVANT TO THEM)
{query_res}
Assistant response:
"""
    def generate():
        for chunk in llm.stream(prompt):
            yield chunk.encode('utf-8')
    
    print(data)
    return Response(generate(), content_type='text/event-stream; charset=utf-8')

if __name__ == '__main__':
    app.run(debug=True, port=5000)