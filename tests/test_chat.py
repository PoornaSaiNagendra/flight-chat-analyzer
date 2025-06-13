import pytest
from fastapi.testclient import TestClient
from app.main import app
import pandas as pd
import json
import os

client = TestClient(app)

@pytest.fixture(scope="module", autouse=True)
def setup_data():
    # Prepare dummy cleaned data
    df = pd.DataFrame({
        "airlie_id": [1],
        "destination": ["NYC"],
        "price": [200]
    })
    df.to_csv("uploaded/cleaned_booking.csv", index=False)

    # Minimal dictionary
    with open("uploaded/data_dictionary.json", "w") as f:
        json.dump({"destination": "City flying to", "price": "Ticket price"}, f)

def test_chat_text_response(monkeypatch):
    from app.services import llm_runner

    # Patch LLM output to inject valid result-producing Python code
    def fake_run_query_prompt(query, df, data_dict):
        return "result = df['destination'].iloc[0]"
    
    monkeypatch.setattr(llm_runner, "run_query_prompt", fake_run_query_prompt)

    res = client.get("/chat/", params={"query": "What is the destination?"})
    assert res.status_code == 200
    assert res.json()["type"] == "text"
    assert res.json()["result"] == "NYC"

def test_chat_graph_response(monkeypatch):
    from app.services import llm_runner
    from app.utils import intent

    # Patch LLM and intent
    def fake_run_query_prompt(query, df, data_dict):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot([1, 2], [3, 4])
        result = "plotted"
        return "import matplotlib.pyplot as plt\nfig, ax = plt.subplots()\nax.plot([1,2],[3,4])\nresult = 'plotted'"
    
    monkeypatch.setattr(llm_runner, "run_query_prompt", fake_run_query_prompt)
    monkeypatch.setattr(intent, "extract_intent", lambda x: "graph")

    res = client.get("/chat/", params={"query": "Show price trend"})
    assert res.status_code == 200
    assert res.json()["type"] == "image"
    assert res.json()["path"].endswith(".png")
    assert os.path.exists(res.json()["path"])
