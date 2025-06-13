from fastapi import APIRouter, Query
import pandas as pd
import json
from app.utils.intent import extract_intent
from app.services.llm_runner import run_query_prompt
from app.services.graph import create_graph_image

router = APIRouter()

@router.get("/")
def chat_with_data(query: str = Query(...)):
    df = pd.read_csv("uploaded/cleaned_booking.csv")
    with open("uploaded/data_dictionary.json") as f:
        data_dict = json.load(f)

    intent = extract_intent(query)
    code = run_query_prompt(query, df, data_dict)

    try:
        local_vars = {"df": df}
        exec(code, {}, local_vars)
        result = local_vars.get("result", "Executed successfully.")

        if intent == "graph":
            path = create_graph_image(local_vars)
            return {"type": "image", "path": path}
        return {"type": "text", "result": str(result)}
    except Exception as e:
        return {"error": str(e)}
