# app/services/llm_runner.py
import os
import json
import pandas as pd
from typing import Dict, Any, List
from app.utils.logger_config import setup_logger
import openai

logger = setup_logger(__name__)

def convert_to_serializable(obj: Any) -> Any:
    """Convert pandas types to Python native types for JSON serialization"""
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, (pd.Int64Dtype, pd.Float64Dtype, pd.StringDtype)):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj

def run_query_prompt(
    prompt: str,
    sample_data: Dict[str, Any] = None,
    context: List[Dict[str, str]] = None,
    max_retries: int = 3
) -> str:
    """
    Run a query prompt with optional sample data and context.
    Handles pandas data types for JSON serialization.
    """
    retry_count = 0
    last_error = None
    
    while retry_count < max_retries:
        try:
            logger.info(f"Attempt {retry_count + 1} of {max_retries}")
            logger.debug(f"Prompt: {prompt}")
            user_message = prompt
            
            if sample_data:
                # Convert pandas types to serializable format
                serializable_data = convert_to_serializable(sample_data)
                logger.debug(f"Sample data: {json.dumps(serializable_data, indent=2)}")
                user_message += f"\n\nHere is the sample data to analyze:\n{json.dumps(serializable_data, indent=2)}"
            
            if context:
                user_message += "\n\nContext from previous interactions:\n"
                for item in context:
                    user_message += f"{item['role']}: {item['content']}\n"
            
            # Add JSON format requirement to the prompt
            user_message += "\n\nPlease provide your analysis in JSON format with the following structure:\n" + \
                           "{\n" + \
                           '  "analysis": {\n' + \
                           '    "column_descriptions": {},\n' + \
                           '    "data_types": {},\n' + \
                           '    "quality_issues": [],\n' + \
                           '    "cleaning_steps": []\n' + \
                           "  }\n" + \
                           "}"
            
            # Initialize Groq client
            client = openai.OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=os.environ.get("GROQ_API_KEY")
            )
            
            logger.info("Initializing Groq client")
            logger.info("Sending request to LLM")
            response = client.chat.completions.create(
                model="qwen/qwen3-32b",
                messages=[
                    {"role": "system", "content": "You are a data analysis assistant that helps analyze flight booking data. Always respond in valid JSON format."},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.1,  # Low temperature for more deterministic responses
                max_tokens=4096,  # Increased token limit for complex responses
                response_format={"type": "json_object"}  # Ensure JSON response
            )
            
            # Log the raw response
            logger.info("Received response from LLM")
            logger.debug(f"Raw LLM response: {response.choices[0].message.content}")
            
            # Parse and validate JSON
            try:
                result = response.choices[0].message.content
                # Validate JSON structure
                json.loads(result)
                logger.info("Successfully validated JSON response")
                return result
            except json.JSONDecodeError as e:
                last_error = e
                logger.warning(f"Invalid JSON response from LLM (attempt {retry_count + 1}): {result}")
                logger.warning(f"JSON decode error: {str(e)}")
                retry_count += 1
                continue
            
        except Exception as e:
            last_error = e
            logger.error(f"Error in LLM query (attempt {retry_count + 1}): {str(e)}", exc_info=True)
            retry_count += 1
            continue
    
    # If we've exhausted all retries
    logger.error(f"Failed to get valid JSON response after {max_retries} attempts")
    if last_error:
        raise ValueError(f"Failed to get valid JSON response after {max_retries} attempts. Last error: {str(last_error)}")
    else:
        raise ValueError(f"Failed to get valid JSON response after {max_retries} attempts")

def convert_pandas_to_json(df: pd.DataFrame) -> Dict[str, Any]:
    """Convert pandas DataFrame to JSON-serializable format"""
    try:
        logger.info("Converting DataFrame to JSON format")
        result = {
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "data": df.astype(str).to_dict(orient='records'),
            "shape": df.shape,
            "summary": {
                "missing_values": df.isnull().sum().to_dict(),
                "unique_counts": df.nunique().to_dict(),
                "numeric_stats": df.describe().to_dict() if not df.empty else {}
            }
        }
        logger.debug(f"Converted DataFrame summary: {json.dumps(result, indent=2)}")
        return result
    except Exception as e:
        logger.error(f"Error converting DataFrame to JSON: {str(e)}", exc_info=True)
        raise
