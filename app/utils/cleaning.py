from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import pandas as pd
import numpy as np

class CleaningAction(BaseModel):
    """Schema for data cleaning actions"""
    action_type: str = Field(description="Type of cleaning action (e.g., 'rename', 'fill_missing', 'convert_type', 'remove_duplicates')")
    column: str = Field(description="Column to apply the action to")
    details: Dict = Field(description="Details of the cleaning action")
    reasoning: str = Field(description="Explanation of why this cleaning is needed")

class DataCleaner:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0)
        self.parser = PydanticOutputParser(pydantic_object=CleaningAction)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at data cleaning and preprocessing.
            Analyze the data and suggest cleaning actions. Consider:
            1. Column naming conventions
            2. Missing value handling
            3. Data type conversions
            4. Duplicate removal
            5. Outlier detection
            6. Inconsistent value standardization
            
            {format_instructions}
            """),
            ("user", """Analyze this data sample and schema:
            Schema: {schema}
            Sample Data: {sample_data}
            """)
        ])

    def analyze_data(self, df: pd.DataFrame) -> List[CleaningAction]:
        """Analyze data and suggest cleaning actions"""
        schema = {col: str(dtype) for col, dtype in df.dtypes.items()}
        sample_data = df.head().to_dict()
        
        prompt = self.prompt.format_messages(
            schema=schema,
            sample_data=sample_data,
            format_instructions=self.parser.get_format_instructions()
        )
        
        response = self.llm(prompt)
        return self.parser.parse(response.content)

    def apply_cleaning(self, df: pd.DataFrame, actions: List[CleaningAction]) -> pd.DataFrame:
        """Apply cleaning actions to the dataframe"""
        cleaned_df = df.copy()
        
        for action in actions:
            if action.action_type == 'rename':
                cleaned_df = cleaned_df.rename(columns={action.column: action.details['new_name']})
            
            elif action.action_type == 'fill_missing':
                if action.details['method'] == 'mean':
                    cleaned_df[action.column] = cleaned_df[action.column].fillna(cleaned_df[action.column].mean())
                elif action.details['method'] == 'median':
                    cleaned_df[action.column] = cleaned_df[action.column].fillna(cleaned_df[action.column].median())
                elif action.details['method'] == 'mode':
                    cleaned_df[action.column] = cleaned_df[action.column].fillna(cleaned_df[action.column].mode()[0])
                elif action.details['method'] == 'value':
                    cleaned_df[action.column] = cleaned_df[action.column].fillna(action.details['value'])
            
            elif action.action_type == 'convert_type':
                cleaned_df[action.column] = cleaned_df[action.column].astype(action.details['new_type'])
            
            elif action.action_type == 'remove_duplicates':
                cleaned_df = cleaned_df.drop_duplicates(subset=action.details.get('subset', None))
        
        return cleaned_df

    def get_cleaning_report(self, actions: List[CleaningAction]) -> str:
        """Generate a human-readable cleaning report"""
        report = "Data Cleaning Report:\n\n"
        
        for action in actions:
            report += f"Action: {action.action_type}\n"
            report += f"Column: {action.column}\n"
            report += f"Details: {action.details}\n"
            report += f"Reasoning: {action.reasoning}\n\n"
        
        return report
