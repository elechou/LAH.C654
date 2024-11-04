from sentence_transformers import SentenceTransformer
from dash import Dash, dcc, html, Input, Output, State, callback
from langchain_chroma import Chroma
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
import os
from dotenv import load_dotenv
import os

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
app = Dash(__name__, title="RAG-ebird", external_stylesheets=external_stylesheets)

# Initialize model
# Load NIM API key
load_dotenv()
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

llm = ChatNVIDIA(model="meta/llama-3.1-70b-instruct")
# Embedding Model
embeddings_model = NVIDIAEmbeddings(
    model="nvidia/llama-3.2-nv-embedqa-1b-v1",
    api_key=NVIDIA_API_KEY,
    truncate="NONE",
)

persist_directory = "./chroma_db"
vectorstore = Chroma(
    persist_directory=persist_directory, embedding_function=embeddings_model
)

template = """  
        Use the following pieces of context to answer the question at the end.
        Sometimes, the user will only give a simple description of the bird,
        or they have meet a bird somewhere, in this case,
        you need to find the most suitable birds from provided context according to user input,
        and express the characteristics of these birds.
        Please only use and describe the bird name inside the context.
        Do not explain birds not mentioned in the provided context!
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Use three sentences maximum and keep the answer as concise as possible.
        
        {context}

        Question: {question}

        Helpful Answer:
        """

custom_rag_prompt = PromptTemplate.from_template(template)
retriever = RunnableLambda(vectorstore.similarity_search_with_relevance_scores).bind(
    k=3
)

retrieved_macaulayID_list = []
similarity_list = []


def format_save_docs(data):
    global retrieved_macaulayID_list
    global similarity_list
    retrieved_macaulayID_list.clear()
    similarity_list.clear()

    formatted_str = "\n\n"
    for doc, score in data:
        retrieved_macaulayID_list.append(doc.metadata["macaulayID"])
        similarity_list.append(score)
        formatted_str += doc.page_content + "\n\n"
    return formatted_str


rag_chain = (
    {"context": retriever | format_save_docs, "question": RunnablePassthrough()}
    | custom_rag_prompt
    | llm
    | StrOutputParser()
)

app.layout = html.Div(
    [
        html.Div(
            [
                dcc.Input(
                    id="input-text-state",
                    type="text",
                    value="Blue Bird by the Pond's Edge.",
                    style={"width": "900px", "margin": "20px"},
                ),
                html.Button(
                    id="submit-button-state",
                    n_clicks=0,
                    children="Search",
                    style={"width": "120px", "margin": "20px", "margin-left": "0"},
                ),
            ],
            style={"display": "flex", "align-items": "center"},
        ),
        dcc.Loading(
            id="loading-1",
            type="default",
            children=html.Div(
                id="rag-output-text",
                style={
                    "width": "1040px",
                    "display": "flex",
                    "flex-direction": "column",
                    "margin": "20px",
                    "font-family": "jura",
                    "font-weight": "bold",
                },
            ),
        ),
        dcc.Loading(
            id="loading-2",
            type="default",
            children=html.Div(
                children=[],
                id="output-state",
                style={"display": "flex", "height": "540px"},
            ),
        ),
        dcc.Store(id="intermediate-data"),
    ],
    style={"display": "flex", "flex-direction": "column", "align-items": "center"},
)


def return_iframe(macaulayID):
    macaulayLink = "https://macaulaylibrary.org/asset/" + macaulayID + "/embed"

    iframeObj = html.Iframe(
        src=macaulayLink,
        height=500,
        width=320,
        style={"border": "none"},
        allow="fullscreen",
    )

    return iframeObj


@callback(
    Output("rag-output-text", "children"),
    Output("intermediate-data", "data"),
    Output("output-state", "children", allow_duplicate=True),
    Input("submit-button-state", "n_clicks"),
    State("input-text-state", "value"),
    prevent_initial_call="initial_duplicate",
)
def update_output(n_clicks, input_text):
    output_text = rag_chain.invoke(input_text)
    global retrieved_macaulayID_list
    global similarity_list

    print(similarity_list)
    print(retrieved_macaulayID_list)

    return output_text, [retrieved_macaulayID_list, similarity_list], []


@callback(
    Output("output-state", "children"),
    Input("intermediate-data", "data"),
)
def update_output(data):
    top_n_macaulayIDs = data[0]
    top_n_similarities = data[1]

    iframes = []
    for macaulayID, similarity in zip(top_n_macaulayIDs, top_n_similarities):
        thisIframe = return_iframe(macaulayID)
        iframes.append(
            html.Div(
                [f"Similarity: {similarity:.4f}", thisIframe],
                style={
                    "display": "flex",
                    "flex-direction": "column",
                    "margin": "20px",
                    "font-family": "jura",
                    "font-weight": "bold",
                },
            )
        )

    return iframes


if __name__ == "__main__":
    app.run(debug=False)
