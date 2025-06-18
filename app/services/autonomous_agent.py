import pandas as pd
import numpy as np
import json
import logging
import os
import base64
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional
from sqlalchemy import create_engine, inspect
from app.services.llm_runner import run_query_prompt, convert_pandas_to_json
from app.utils.logger_config import setup_logger

logger = setup_logger(__name__)

class AutonomousAgent:
    def __init__(self):
        """Initialize the autonomous agent"""
        self.engine = create_engine('sqlite:///:memory:')
        self.chat_history = {}
        self.column_descriptions = {}
        self.data_quality = {}
        logger.info("AutonomousAgent initialized with in-memory SQLite database")

    def initialize_database(self):
        """Initialize in-memory SQLite database"""
        try:
            logger.info("Initializing in-memory database")
            # The engine is already created in __init__
            return self.engine
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}", exc_info=True)
            raise

    def analyze_columns(self, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """Analyze columns in a dataset using LLM"""
        try:
            logger.info(f"Analyzing columns for {dataset_name}")
            
            # Get sample data
            sample_data = {
                "columns": df.columns.tolist(),
                "dtypes": df.dtypes.astype(str).to_dict(),
                "sample": df.head(5).to_dict(orient='records'),
                "missing_values": df.isnull().sum().to_dict(),
                "unique_counts": df.nunique().to_dict()
            }
            
            # Create analysis prompt
            prompt = f"""Analyze the following {dataset_name} dataset and provide:
            1. Column descriptions
            2. Data types and their implications
            3. Potential data quality issues
            4. Suggested cleaning steps
            
            Focus on:
            - Business meaning of each column
            - Data quality and completeness
            - Potential relationships with other columns
            - Any anomalies or patterns in the data
            """
            
            # Get LLM analysis
            analysis = run_query_prompt(
                prompt=prompt,
                sample_data=sample_data
            )
            
            return {
                "dataset": dataset_name,
                "analysis": analysis,
                "metadata": sample_data
            }
            
        except Exception as e:
            logger.error(f"Error analyzing columns: {str(e)}", exc_info=True)
            raise

    def process_files(self, booking_path: str, airline_path: str, session_id: str) -> Dict[str, Any]:
        """Process booking and airline files"""
        try:
            logger.info(f"Processing files for session {session_id}")
            
            # Read files
            booking_df = pd.read_csv(booking_path)
            airline_df = pd.read_csv(airline_path)
            
            # Clean data
            booking_df = self.clean_data(booking_df)
            airline_df = self.clean_data(airline_df)
            
            # Generate column descriptions
            self.column_descriptions = self.generate_column_descriptions(booking_df, airline_df)
            
            # Analyze data quality
            data_quality = self.analyze_data_quality(booking_df)
            
            # Load data into SQLite
            booking_df.to_sql('bookings', self.engine, if_exists='replace', index=False)
            airline_df.to_sql('airline_mapping', self.engine, if_exists='replace', index=False)
            
            logger.info("Files processed successfully")
            
            return {
                "data_quality": data_quality,
                "column_descriptions": self.column_descriptions
            }
            
        except Exception as e:
            logger.error(f"Error processing files: {str(e)}", exc_info=True)
            raise

    def process_query(self, query: str, session_id: str) -> Dict[str, Any]:
        """Process a natural language query"""
        try:
            logger.info(f"Processing query for session {session_id}")
            logger.info(f"Query: {query}")
            
            # Analyze query intent
            intent_prompt = """Analyze this query and determine if it's asking for insights or visualizations. 
            Your response MUST be in this exact JSON format:
            {
                "intent": "insights|visualization",
                "reason": "explanation of why this intent was chosen",
                "required_columns": ["list", "of", "needed", "columns"]
            }"""
            
            logger.info("Getting query intent from LLM")
            intent_response = run_query_prompt(intent_prompt, {"query": query})
            
            intent_data = json.loads(intent_response)
            logger.info(f"Query intent: {intent_data['intent']}")
            logger.info(f"Intent reason: {intent_data.get('reason', 'No reason provided')}")
            logger.info(f"Required columns: {intent_data.get('required_columns', [])}")
            
            # Get schema information
            schema_info = self.get_schema_info()
            logger.info(f"Schema information: {json.dumps(schema_info, indent=2)}")
            
            # Generate and execute SQL query with retries
            max_retries = 3
            retry_count = 0
            last_error = None
            result_df = None
            
            while retry_count < max_retries:
                try:
                    # Generate SQL query
                    sql_prompt = f"""Convert this query to SQL: {query}
                    Available tables: {self._get_table_info()}
                    Schema information: {schema_info}
                    {f'Previous error: {last_error}' if last_error else ''}
                    Your response MUST be in this exact JSON format:
                    {{
                        "sql": "the SQL query",
                        "explanation": "explanation of what the query does",
                        "expected_columns": ["list", "of", "columns", "in", "result"]
                    }}"""
                    
                    logger.info(f"Generating SQL query (attempt {retry_count + 1})")
                    sql_response = run_query_prompt(sql_prompt)
                    
                    try:
                        sql_data = json.loads(sql_response)
                        logger.info(f"Raw SQL response: {sql_response}")
                        
                        # Validate required fields
                        if not isinstance(sql_data, dict):
                            raise ValueError("SQL response is not a dictionary")
                            
                        if "sql" not in sql_data:
                            # Try to extract SQL from the response if it's not in the expected format
                            if isinstance(sql_response, str) and "SELECT" in sql_response.upper():
                                sql_query = sql_response[sql_response.upper().find("SELECT"):].strip()
                                sql_data = {
                                    "sql": sql_query,
                                    "explanation": "Extracted SQL from response",
                                    "expected_columns": []
                                }
                            else:
                                raise ValueError("No SQL query found in response")
                        
                        sql_query = sql_data["sql"].strip()
                        if not sql_query.lower().startswith("select"):
                            raise ValueError("Only SELECT queries are allowed")
                        
                        logger.info(f"Generated SQL: {sql_query}")
                        logger.info(f"SQL explanation: {sql_data.get('explanation', 'No explanation provided')}")
                        logger.info(f"Expected columns: {sql_data.get('expected_columns', [])}")
                        
                        # Execute SQL query
                        logger.info("Executing SQL query")
                        result_df = pd.read_sql_query(sql_query, self.engine)
                        logger.info(f"Query returned {len(result_df)} rows")
                        logger.info(f"Query result columns: {result_df.columns.tolist()}")
                        logger.info(f"Query result sample: {result_df.head().to_dict(orient='records')}")
                        
                        # If we get here, the query was successful
                        break
                        
                    except json.JSONDecodeError as e:
                        last_error = f"Invalid JSON response: {str(e)}"
                        logger.info(f"SQL response parsing failed (attempt {retry_count + 1}): {last_error}")
                        retry_count += 1
                        if retry_count == max_retries:
                            raise ValueError(f"Failed to generate valid SQL query after {max_retries} attempts. Last error: {last_error}")
                        continue
                    except Exception as e:
                        last_error = str(e)
                        logger.info(f"SQL query failed (attempt {retry_count + 1}): {last_error}")
                        retry_count += 1
                        if retry_count == max_retries:
                            raise ValueError(f"Failed to generate valid SQL query after {max_retries} attempts. Last error: {last_error}")
                        continue
                except Exception as e:
                    last_error = str(e)
                    logger.info(f"SQL generation failed (attempt {retry_count + 1}): {last_error}")
                    retry_count += 1
                    if retry_count == max_retries:
                        raise ValueError(f"Failed to generate valid SQL query after {max_retries} attempts. Last error: {last_error}")
                    continue
            
            if result_df is None:
                raise ValueError("Failed to execute SQL query after all retries")
            
            # Generate visualization if requested
            if intent_data["intent"] == "visualization":
                logger.info("Generating visualization")
                viz_prompt = f"""Create a visualization for this data:
                {convert_pandas_to_json(result_df)}
                Query: {query}
                Your response MUST be in this exact JSON format:
                {{
                    "code": "the Python code to create the visualization",
                    "type": "bar|line|scatter|pie|heatmap",
                    "explanation": "explanation of the visualization"
                }}"""
                
                viz_response = run_query_prompt(viz_prompt)
                
                viz_data = json.loads(viz_response)
                logger.info(f"Generated visualization type: {viz_data['type']}")
                logger.info(f"Visualization explanation: {viz_data.get('explanation', 'No explanation provided')}")
                
                # Execute visualization code
                try:
                    logger.info("Executing visualization code")
                    namespace = {"df": result_df, "plt": plt, "sns": sns}
                    exec(viz_data["code"], namespace)
                    plt.savefig("temp_viz.png")
                    plt.close()
                    
                    # Convert plot to base64
                    with open("temp_viz.png", "rb") as f:
                        viz_base64 = base64.b64encode(f.read()).decode()
                    os.remove("temp_viz.png")
                    
                    logger.info("Visualization generated successfully")
                    response = {
                        "type": "visualization",
                        "response": viz_data["explanation"],
                        "data": viz_base64
                    }
                    logger.info(f"Visualization response structure: {json.dumps({k: v[:100] + '...' if isinstance(v, str) and len(v) > 100 else v for k, v in response.items()}, indent=2)}")
                    return response
                    
                except Exception as e:
                    logger.error(f"Error generating visualization: {str(e)}")
                    raise ValueError(f"Error generating visualization: {str(e)}")
            
            # Generate insights
            logger.info("Generating insights")
            insights_prompt = f"""Analyze this data and provide insights:
            {convert_pandas_to_json(result_df)}
            Query: {query}
            Your response MUST be in this exact JSON format:
            {{
                "insights": [
                    {{
                        "title": "insight title",
                        "description": "detailed insight",
                        "importance": "high|medium|low"
                    }}
                ],
                "summary": "overall summary of findings"
            }}"""
            
            insights_response = run_query_prompt(insights_prompt)
            
            try:
                insights_data = json.loads(insights_response)
                logger.info(f"Parsed insights data: {json.dumps(insights_data, indent=2)}")
                
                # Validate insights response format
                if not isinstance(insights_data, dict):
                    raise ValueError("Insights response is not a dictionary")
                
                if "insights" not in insights_data or not isinstance(insights_data["insights"], list):
                    # If insights are missing or invalid, create a default structure
                    insights_data = {
                        "insights": [{
                            "title": "Data Analysis",
                            "description": str(result_df.describe().to_dict()),
                            "importance": "high"
                        }],
                        "summary": "Analysis of the query results"
                    }
                    logger.info("Created default insights structure due to invalid response")
                
                if "summary" not in insights_data:
                    insights_data["summary"] = "Analysis of the query results"
                    logger.info("Added default summary due to missing field")
                
                logger.info(f"Generated {len(insights_data['insights'])} insights")
                logger.info(f"Insights summary: {insights_data['summary']}")
                logger.info(f"Insights details: {json.dumps(insights_data['insights'], indent=2)}")
                
                response = {
                    "type": "insights",
                    "response": insights_data["summary"],
                    "data": insights_data["insights"]
                }
                logger.info(f"Insights response structure: {json.dumps(response, indent=2)}")
                return response
                
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing insights response: {str(e)}")
                # Return a default response if JSON parsing fails
                response = {
                    "type": "insights",
                    "response": "Analysis of the query results",
                    "data": [{
                        "title": "Data Analysis",
                        "description": str(result_df.describe().to_dict()),
                        "importance": "high"
                    }]
                }
                logger.info(f"Returning default response due to JSON parsing error: {json.dumps(response, indent=2)}")
                return response
            except Exception as e:
                logger.error(f"Error processing insights: {str(e)}")
                raise
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            raise

    def _determine_query_intent(self, query: str, context: Dict) -> Dict:
        """Determine the intent of the query"""
        try:
            logger.info("Determining query intent")
            
            # Prepare context with conversation history
            conversation_history = self.chat_history.get_conversation_history(context["session_id"], limit=5)
            
            # Get intent from LLM
            intent = run_query_prompt(
                query="""Analyze this query and determine:
                1. If it needs visualization
                2. What type of visualization
                3. Required columns and aggregations
                4. Time series or comparison needs
                5. Suggest follow-up questions""",
                schema=self.schemas["query_intent"],
                sample_data={
                    "query": query,
                    "conversation_history": conversation_history,
                    "column_descriptions": context.get("column_descriptions", {}),
                    "data_summary": context.get("data_summary", {})
                }
            )
            
            return json.loads(intent)
            
        except Exception as e:
            logger.error(f"Error determining query intent: {str(e)}", exc_info=True)
            raise

    def _generate_sql(self, query: str, intent: Dict, context: Dict) -> Dict:
        """Generate SQL query from natural language"""
        try:
            logger.info("Generating SQL query")
            
            # Prepare context with conversation history
            conversation_history = self.chat_history.get_conversation_history(context["session_id"], limit=5)
            
            # Get SQL from LLM
            sql_result = run_query_prompt(
                query="""Convert this natural language query to SQL:
                1. Use appropriate joins and aggregations
                2. Handle edge cases and null values
                3. Optimize for performance
                4. Include explanation of the query
                5. Suggest related queries""",
                schema=self.schemas["sql_generation"],
                sample_data={
                    "query": query,
                    "intent": intent,
                    "conversation_history": conversation_history,
                    "column_descriptions": context.get("column_descriptions", {}),
                    "tables": ["bookings", "airlines"]
                }
            )
            
            return json.loads(sql_result)
            
        except Exception as e:
            logger.error(f"Error generating SQL: {str(e)}", exc_info=True)
            raise

    def _generate_visualization(self, df: pd.DataFrame, intent: Dict, query: str, context: Dict) -> Dict:
        """Generate visualization code and execute it"""
        try:
            logger.info("Generating visualization")
            
            # Prepare context with conversation history
            conversation_history = self.chat_history.get_conversation_history(context["session_id"], limit=5)
            
            # Get visualization code from LLM
            viz_code = run_query_prompt(
                query="""Generate Python code to create a visualization that:
                1. Matches the query intent
                2. Uses appropriate chart type
                3. Includes proper labels and formatting
                4. Handles edge cases
                5. Suggests follow-up questions""",
                schema=self.schemas["visualization_code"],
                sample_data={
                    "data_sample": df.head().to_dict(),
                    "columns": list(df.columns),
                    "intent": intent,
                    "query": query,
                    "conversation_history": conversation_history,
                    "visualization_history": context.get("visualization_history", [])
                }
            )
            
            # Execute visualization code
            viz_result = json.loads(viz_code)
            namespace = {
                "plt": plt,
                "sns": sns,
                "pd": pd,
                "np": np,
                "df": df
            }
            
            # Create figure
            plt.figure(figsize=(12, 8))
            exec(viz_result["code"], namespace)
            
            # Save to buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            image_base64 = base64.b64encode(buf.getvalue()).decode()
            plt.close()
            
            return {
                "image": image_base64,
                "explanation": viz_result["explanation"],
                "suggested_follow_up": viz_result.get("suggested_follow_up", [])
            }
            
        except Exception as e:
            logger.error(f"Error generating visualization: {str(e)}", exc_info=True)
            raise

    def _generate_insights(self, df: pd.DataFrame, query: str, intent: Dict, context: Dict) -> List[Dict]:
        """Generate insights from query results"""
        try:
            logger.info("Generating insights")
            
            # Prepare context with conversation history
            conversation_history = self.chat_history.get_conversation_history(context["session_id"], limit=5)
            
            # Get insights from LLM
            insights = run_query_prompt(
                query="""Analyze this data and provide:
                1. Key findings and patterns
                2. Business implications
                3. Notable trends or anomalies
                4. Recommendations
                5. Suggest follow-up questions""",
                schema={
                    "type": "object",
                    "properties": {
                        "insights": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "finding": {"type": "string"},
                                    "significance": {"type": "string"},
                                    "supporting_data": {"type": "string"}
                                }
                            }
                        },
                        "follow_up_questions": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    }
                },
                sample_data={
                    "data": df.to_dict(orient="records"),
                    "query": query,
                    "intent": intent,
                    "conversation_history": conversation_history,
                    "column_descriptions": context.get("column_descriptions", {})
                }
            )
            
            return json.loads(insights)
            
        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}", exc_info=True)
            raise

    def get_chat_history(self, session_id: str, limit: Optional[int] = None) -> List[Dict]:
        """Get chat history for a session"""
        return self.chat_history.get_conversation_history(session_id, limit)

    def clear_chat_history(self, session_id: str) -> None:
        """Clear chat history for a session"""
        self.chat_history.clear_conversation(session_id)

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean data based on LLM analysis with retry mechanism"""
        try:
            logger.info("Getting cleaning plan from LLM")
            
            # Prepare sample data
            sample_data = {
                "columns": df.columns.tolist(),
                "dtypes": df.dtypes.astype(str).to_dict(),
                "sample": df.head(5).to_dict(orient='records'),
                "missing_values": df.isnull().sum().to_dict(),
                "unique_counts": df.nunique().to_dict()
            }
            
            # Create cleaning prompt
            prompt = """Analyze this data and provide a cleaning plan. Your response MUST be in this exact JSON format:
            {
                "cleaning_steps": [
                    {
                        "column": "column_name",
                        "action": "fill_missing|convert_type|remove_duplicates|standardize",
                        "reason": "explanation",
                        "target_type": "int|float|datetime|string" (only if action is convert_type)
                    }
                ],
                "data_quality_issues": [
                    {
                        "issue": "description",
                        "impact": "impact_description",
                        "solution": "proposed_solution"
                    }
                ]
            }

            Focus on:
            1. Handling missing values
            2. Correcting data types
            3. Removing duplicates
            4. Standardizing formats
            5. Handling outliers

            IMPORTANT: Your response MUST be valid JSON and MUST include the 'cleaning_steps' array.
            """
            
            try:
                # Get cleaning plan from LLM with retries
                cleaning_plan = run_query_prompt(
                    prompt=prompt,
                    sample_data=sample_data,
                    max_retries=3
                )
                
                # Parse cleaning plan
                plan = json.loads(cleaning_plan)
                
                # Validate plan structure
                if not isinstance(plan, dict):
                    raise ValueError("LLM response is not a dictionary")
                
                if "cleaning_steps" not in plan:
                    raise ValueError("Missing 'cleaning_steps' in LLM response")
                
            except Exception as e:
                logger.error(f"Failed to get valid cleaning plan after retries: {str(e)}")
                logger.info("Using fallback cleaning steps")
                # Create fallback cleaning steps
                plan = {
                    "cleaning_steps": [],
                    "data_quality_issues": []
                }
                
                # Add basic cleaning steps for each column
                for col in df.columns:
                    # Handle missing values
                    if df[col].isnull().any():
                        plan["cleaning_steps"].append({
                            "column": col,
                            "action": "fill_missing",
                            "reason": "Handling missing values"
                        })
                    
                    # Handle data types
                    if df[col].dtype == 'object':
                        plan["cleaning_steps"].append({
                            "column": col,
                            "action": "standardize",
                            "reason": "Standardizing string values"
                        })
                    elif pd.api.types.is_numeric_dtype(df[col]):
                        plan["cleaning_steps"].append({
                            "column": col,
                            "action": "convert_type",
                            "reason": "Ensuring numeric type",
                            "target_type": "float"
                        })
            
            # Apply cleaning steps
            logger.info("Applying cleaning steps")
            for step in plan["cleaning_steps"]:
                try:
                    column = step.get("column")
                    if not column or column not in df.columns:
                        logger.warning(f"Invalid column name in cleaning step: {step}")
                        continue
                        
                    action = step.get("action")
                    if not action:
                        logger.warning(f"Missing action in cleaning step: {step}")
                        continue
                    
                    logger.info(f"Applying {action} to column {column}")
                    
                    if action == "fill_missing":
                        df[column] = df[column].fillna(df[column].mode()[0] if not df[column].mode().empty else None)
                    elif action == "convert_type":
                        target_type = step.get("target_type", "").lower()
                        if "int" in target_type:
                            df[column] = pd.to_numeric(df[column], errors='coerce').fillna(0).astype(int)
                        elif "float" in target_type:
                            df[column] = pd.to_numeric(df[column], errors='coerce')
                        elif "datetime" in target_type:
                            df[column] = pd.to_datetime(df[column], errors='coerce')
                    elif action == "remove_duplicates":
                        df = df.drop_duplicates(subset=[column])
                    elif action == "standardize":
                        if df[column].dtype == 'object':
                            df[column] = df[column].astype(str).str.lower().str.strip()
                    
                except Exception as e:
                    logger.error(f"Error applying cleaning step {step}: {str(e)}")
                    continue
            
            logger.info("Data cleaning completed")
            return df
            
        except Exception as e:
            logger.error(f"Error cleaning data: {str(e)}", exc_info=True)
            raise

    def analyze_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data quality using LLM"""
        try:
            logger.info("Starting data quality analysis")
            
            # Prepare sample data
            sample_data = {
                "columns": df.columns.tolist(),
                "dtypes": df.dtypes.astype(str).to_dict(),
                "sample": df.head(5).to_dict(orient='records'),
                "missing_values": df.isnull().sum().to_dict(),
                "unique_counts": df.nunique().to_dict()
            }
            logger.debug(f"Sample data for analysis: {json.dumps(sample_data, indent=2)}")
            
            # Create analysis prompt
            prompt = """Analyze this data and provide a quality assessment. Your response MUST be in this exact JSON format:
            {
                "data_quality_issues": [
                    {
                        "issue": "description of the issue",
                        "impact": "impact on analysis",
                        "solution": "proposed solution"
                    }
                ],
                "recommendations": [
                    {
                        "action": "specific action to take",
                        "priority": "high|medium|low",
                        "reason": "explanation"
                    }
                ],
                "quality_score": {
                    "overall": 0-100,
                    "completeness": 0-100,
                    "consistency": 0-100,
                    "accuracy": 0-100
                },
                "summary": "overall assessment"
            }"""
            
            logger.info("Getting data quality analysis from LLM")
            analysis_response = run_query_prompt(prompt, sample_data)
            logger.debug(f"Analysis response: {analysis_response}")
            
            analysis_data = json.loads(analysis_response)
            
            # Log quality metrics
            if "quality_score" in analysis_data:
                logger.info(f"Data quality scores: {analysis_data['quality_score']}")
            logger.info(f"Found {len(analysis_data.get('data_quality_issues', []))} issues")
            logger.info(f"Generated {len(analysis_data.get('recommendations', []))} recommendations")
            
            # Ensure all required fields are present
            if "data_quality_issues" not in analysis_data:
                analysis_data["data_quality_issues"] = []
            if "recommendations" not in analysis_data:
                analysis_data["recommendations"] = []
            if "quality_score" not in analysis_data:
                analysis_data["quality_score"] = {
                    "overall": 0,
                    "completeness": 0,
                    "consistency": 0,
                    "accuracy": 0
                }
            if "summary" not in analysis_data:
                analysis_data["summary"] = "No summary provided"
            
            return analysis_data
            
        except Exception as e:
            logger.error(f"Error analyzing data quality: {str(e)}", exc_info=True)
            # Return default structure in case of error
            return {
                "data_quality_issues": [],
                "recommendations": [],
                "quality_score": {
                    "overall": 0,
                    "completeness": 0,
                    "consistency": 0,
                    "accuracy": 0
                },
                "summary": f"Error during analysis: {str(e)}"
            }

    def get_schema_info(self) -> Dict[str, Any]:
        """Get schema information for all tables"""
        try:
            logger.info("Getting schema information")
            schema_info = {}
            
            # Get list of all tables
            tables = pd.read_sql_query(
                "SELECT name FROM sqlite_master WHERE type='table'",
                self.engine
            )['name'].tolist()
            
            for table in tables:
                # Get column information
                columns = pd.read_sql_query(
                    f"PRAGMA table_info({table})",
                    self.engine
                )
                
                # Get sample data
                sample = pd.read_sql_query(
                    f"SELECT * FROM {table} LIMIT 5",
                    self.engine
                )
                
                # Get column statistics
                stats = {}
                for col in sample.columns:
                    col_stats = {
                        "type": str(sample[col].dtype),
                        "unique_values": int(sample[col].nunique()),
                        "missing_values": int(sample[col].isnull().sum()),
                        "sample_values": [str(x) for x in sample[col].head(3).tolist()]
                    }
                    if pd.api.types.is_numeric_dtype(sample[col]):
                        col_stats.update({
                            "min": float(sample[col].min()),
                            "max": float(sample[col].max()),
                            "mean": float(sample[col].mean())
                        })
                    stats[col] = col_stats
                
                # Convert row count to int
                row_count = int(pd.read_sql_query(
                    f"SELECT COUNT(*) as count FROM {table}",
                    self.engine
                )['count'].iloc[0])
                
                schema_info[table] = {
                    "columns": columns.to_dict(orient='records'),
                    "statistics": stats,
                    "row_count": row_count
                }
            
            logger.info(f"Retrieved schema information for {len(tables)} tables")
            return schema_info
            
        except Exception as e:
            logger.error(f"Error getting schema information: {str(e)}", exc_info=True)
            raise

    def generate_column_descriptions(self, booking_df: pd.DataFrame, airline_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate descriptions for columns in both datasets"""
        try:
            logger.info("Generating column descriptions")
            
            # Prepare sample data
            sample_data = {
                "booking_columns": {
                    "columns": booking_df.columns.tolist(),
                    "dtypes": booking_df.dtypes.astype(str).to_dict(),
                    "sample": booking_df.head(5).to_dict(orient='records'),
                    "missing_values": booking_df.isnull().sum().to_dict(),
                    "unique_counts": booking_df.nunique().to_dict()
                },
                "airline_columns": {
                    "columns": airline_df.columns.tolist(),
                    "dtypes": airline_df.dtypes.astype(str).to_dict(),
                    "sample": airline_df.head(5).to_dict(orient='records'),
                    "missing_values": airline_df.isnull().sum().to_dict(),
                    "unique_counts": airline_df.nunique().to_dict()
                }
            }
            
            # Create description prompt
            prompt = """Analyze these datasets and provide detailed descriptions for each column. 
            Your response MUST be in this exact JSON format:
            {
                "booking_columns": {
                    "column_name": {
                        "description": "detailed description",
                        "data_type": "type of data",
                        "business_meaning": "business context",
                        "relationships": ["related columns"],
                        "quality_notes": "data quality observations"
                    }
                },
                "airline_columns": {
                    "column_name": {
                        "description": "detailed description",
                        "data_type": "type of data",
                        "business_meaning": "business context",
                        "relationships": ["related columns"],
                        "quality_notes": "data quality observations"
                    }
                }
            }

            Focus on:
            1. Business meaning of each column
            2. Data types and their implications
            3. Relationships between columns
            4. Data quality observations
            5. Potential use cases
            """
            
            # Get descriptions from LLM
            descriptions_response = run_query_prompt(
                prompt=prompt,
                sample_data=sample_data
            )
            
            # Parse and validate response
            descriptions = json.loads(descriptions_response)
            logger.info("Successfully generated column descriptions")
            
            return descriptions
            
        except Exception as e:
            logger.error(f"Error generating column descriptions: {str(e)}", exc_info=True)
            raise

    def _get_table_info(self) -> Dict[str, Any]:
        """Get information about all tables in the database"""
        try:
            inspector = inspect(self.engine)
            tables = inspector.get_table_names()
            
            table_info = {}
            for table in tables:
                columns = inspector.get_columns(table)
                table_info[table] = {
                    "columns": [col["name"] for col in columns],
                    "primary_keys": inspector.get_pk_constraint(table)["constrained_columns"],
                    "foreign_keys": [
                        {
                            "referred_table": fk["referred_table"],
                            "referred_columns": fk["referred_columns"],
                            "constrained_columns": fk["constrained_columns"]
                        }
                        for fk in inspector.get_foreign_keys(table)
                    ]
                }
            
            return table_info
            
        except Exception as e:
            logger.error(f"Error getting table info: {str(e)}", exc_info=True)
            return {} 