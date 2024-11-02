from sentence_transformers import SentenceTransformer
import os
import chromadb
from dash import Dash, dcc, html, Input, Output, State, callback

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
app = Dash(__name__, title="RAG-ebird", external_stylesheets=external_stylesheets)

# Initialize model
cache_folder = os.path.expanduser("/Users/shou/Code/huggingface_models")
persist_dir = os.path.expanduser("./chroma_db")

if "model" not in globals():
    print("Loading dunzhang/stella_en_1.5B_v5...")
    model = SentenceTransformer(
        "dunzhang/stella_en_1.5B_v5",
        cache_folder=cache_folder,
        local_files_only=False,
        trust_remote_code=True,
    )

# Create ChromaDB client
client = chromadb.PersistentClient(path=persist_dir)  # Directory for data persistence

# Get collection
collection = client.get_collection(name="bird_entries")
print("Database bird_entries connected...")


# Query function
def match(query, top_k=3):
    """
    # Usage
    # top_n_similarities, top_n_macaulayIDs = match("Blue Bird by the Pond's Edge.")
    # print("\nQuery results:", top_n_similarities)
    # print("\nQuery results:", top_n_macaulayIDs)
    """
    query_prompt_name = "s2p_query"
    query_embedding = model.encode(query, prompt_name=query_prompt_name)

    # Query using ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding.tolist()], n_results=top_k
    )

    # Format results
    similarities = {}
    macaulayIDs = {}
    for id, distance, metadata in zip(
        results["ids"][0], results["distances"][0], results["metadatas"][0]
    ):
        similarities[id] = 1 - distance
        macaulayIDs[id] = metadata["macaulayID"]

    return similarities, macaulayIDs


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
            id="loading",
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
    Output("intermediate-data", "data"),
    Output("output-state", "children", allow_duplicate=True),
    Input("submit-button-state", "n_clicks"),
    State("input-text-state", "value"),
    prevent_initial_call='initial_duplicate',
)
def update_output(n_clicks, input_text):
    top_n_similarities, top_n_macaulayIDs = match(input_text)
    intermediate = {"similarities": top_n_similarities, "macaulayIDs": top_n_macaulayIDs}

    return intermediate, []

@callback(
    Output("output-state", "children"),
    Input("intermediate-data", "data"),
)
def update_output(data):
    top_n_similarities = data["similarities"]
    top_n_macaulayIDs = data["macaulayIDs"]
    
    iframes = []
    for key, similarity in top_n_similarities.items():
        thisIframe = return_iframe(top_n_macaulayIDs[key])
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
