from openai import OpenAI
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import io
import base64
import os
from .intent import IntentClassifier
from .cleaning import DataCleaner

def generate_data_dictionary(df):
    sample = df.head(5).to_csv(index=False, sep='\t')
    message = f"""
    Based on below column names and first 5 values generate description for each column:
    {sample}
    Give output in json format: {{<column>: <description>}}
    """
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.getenv("GROQ_API_KEY")
    )
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You're a helpful assistant"},
            {"role": "user", "content": message}
        ],
        model="mixtral-8x7b-32768"
    )
    return response.choices[0].message.content

class LLMAnalyzer:
    def __init__(self):
        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv("GROQ_API_KEY")
        )
        self.intent_classifier = IntentClassifier()
        self.data_cleaner = DataCleaner()
        
        # SQL generation prompt
        self.sql_system_prompt = """You are an expert at converting natural language queries to SQL.
        Consider the database schema and generate optimized SQL queries.
        Always include appropriate JOINs and WHERE clauses.
        Use proper indexing hints when available."""
        
        # Visualization prompt
        self.viz_system_prompt = """You are an expert at data visualization.
        Given the data and query context, suggest the best visualization approach.
        Consider:
        1. Data type and distribution
        2. Number of variables to show
        3. Key insights to highlight
        4. Appropriate chart type and styling"""

    def analyze_query(self, query: str, schema: Dict, sample_data: Dict) -> Dict[str, Any]:
        """Analyze query and determine appropriate response type"""
        # Classify intent
        intent = self.intent_classifier.classify_intent(query)
        
        # Generate SQL
        sql_response = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": self.sql_system_prompt},
                {"role": "user", "content": f"Schema: {schema}\nSample Data: {sample_data}\nConvert this query to SQL: {query}"}
            ],
            model="mixtral-8x7b-32768"
        )
        sql_result = sql_response.choices[0].message.content
        
        response = {
            "sql_query": sql_result,
            "needs_visualization": intent.needs_visualization,
            "visualization_type": intent.visualization_type,
            "reasoning": intent.reasoning
        }
        
        return response

    def generate_visualization(self, data: pd.DataFrame, query_context: str) -> Optional[str]:
        """Generate visualization based on data and context"""
        # Get visualization parameters
        viz_response = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": self.viz_system_prompt},
                {"role": "user", "content": f"Data: {data.to_dict()}\nQuery Context: {query_context}\nSuggest visualization parameters for this data"}
            ],
            model="mixtral-8x7b-32768"
        )
        viz_params = eval(viz_response.choices[0].message.content)
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        
        if viz_params['type'] == 'bar':
            sns.barplot(data=data, x=viz_params['x'], y=viz_params['y'])
        elif viz_params['type'] == 'line':
            sns.lineplot(data=data, x=viz_params['x'], y=viz_params['y'])
        elif viz_params['type'] == 'scatter':
            sns.scatterplot(data=data, x=viz_params['x'], y=viz_params['y'])
        elif viz_params['type'] == 'heatmap':
            sns.heatmap(data=data, annot=True, cmap='YlOrRd')
        
        plt.title(viz_params.get('title', 'Data Visualization'))
        plt.tight_layout()
        
        # Convert plot to base64 string
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode()
        plt.close()
        
        return img_str

    def clean_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
        """Clean data using LLM-powered analysis"""
        # Get cleaning actions
        actions = self.data_cleaner.analyze_data(df)
        
        # Apply cleaning
        cleaned_df = self.data_cleaner.apply_cleaning(df, actions)
        
        # Generate report
        report = self.data_cleaner.get_cleaning_report(actions)
        
        return cleaned_df, report
