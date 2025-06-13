from fastapi import APIRouter, UploadFile, File
import pandas as pd
import os
from app.utils.cleaning import clean_and_merge_files
from app.utils.llm_analysis import generate_data_dictionary

router = APIRouter()

UPLOAD_DIR = "uploaded"
BOOKING_PATH = os.path.join(UPLOAD_DIR, "booking.csv")
AIRLINE_PATH = os.path.join(UPLOAD_DIR, "airlines.csv")
CLEANED_PATH = os.path.join(UPLOAD_DIR, "cleaned_booking.csv")
DATA_DICT_PATH = os.path.join(UPLOAD_DIR, "data_dictionary.json")
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/")
async def upload_files(booking_file: UploadFile = File(...), airline_file: UploadFile = File(...)):
    with open(BOOKING_PATH, "wb") as f:
        f.write(await booking_file.read())

    with open(AIRLINE_PATH, "wb") as f:
        f.write(await airline_file.read())

    booking_df = pd.read_csv(BOOKING_PATH)
    airline_df = pd.read_csv(AIRLINE_PATH)

    df_cleaned = clean_and_merge_files(booking_df, airline_df)
    df_cleaned.to_csv(CLEANED_PATH, index=False)

    data_dict = generate_data_dictionary(df_cleaned)
    with open(DATA_DICT_PATH, "w") as f:
        f.write(data_dict)

    return {"message": "Files uploaded and processed successfully."}
