from langchain_core.messages import HumanMessage, SystemMessage
from app.services.llm_runner import chat_model
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import io
import base64
from .intent import IntentClassifier
from .cleaning import DataCleaner

def generate_data_dictionary(df):
    sample = df.head(5).to_csv(index=False, sep='\t')
    message = f"""
    Based on below column names and first 5 values generate description for each column:
    {sample}
    Give output in json format: {{<column>: <description>}}
    """
    response = chat_model.invoke([SystemMessage(content="You're a helpful assistant"), HumanMessage(content=message)])
    return response.content

class LLMAnalyzer:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0)
        self.intent_classifier = IntentClassifier()
        self.data_cleaner = DataCleaner()
        
        # SQL generation prompt
        self.sql_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at converting natural language queries to SQL.
            Consider the database schema and generate optimized SQL queries.
            Always include appropriate JOINs and WHERE clauses.
            Use proper indexing hints when available.
            
            Schema: {schema}
            Sample Data: {sample_data}
            """),
            ("user", "Convert this query to SQL: {query}")
        ])
        
        # Visualization prompt
        self.viz_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at data visualization.
            Given the data and query context, suggest the best visualization approach.
            Consider:
            1. Data type and distribution
            2. Number of variables to show
            3. Key insights to highlight
            4. Appropriate chart type and styling
            
            Data: {data}
            Query Context: {query_context}
            """),
            ("user", "Suggest visualization parameters for this data")
        ])

    def analyze_query(self, query: str, schema: Dict, sample_data: Dict) -> Dict[str, Any]:
        """Analyze query and determine appropriate response type"""
        # Classify intent
        intent = self.intent_classifier.classify_intent(query)
        
        # Generate SQL
        sql_chain = LLMChain(llm=self.llm, prompt=self.sql_prompt)
        sql_result = sql_chain.run(
            query=query,
            schema=schema,
            sample_data=sample_data
        )
        
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
        viz_chain = LLMChain(llm=self.llm, prompt=self.viz_prompt)
        viz_params = viz_chain.run(
            data=data.to_dict(),
            query_context=query_context
        )
        
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
