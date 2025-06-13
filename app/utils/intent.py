from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional
import json

class VisualizationIntent(BaseModel):
    """Schema for visualization intent classification"""
    needs_visualization: bool = Field(description="Whether the query requires visualization")
    visualization_type: Optional[str] = Field(description="Type of visualization if needed (e.g., 'bar', 'line', 'scatter', 'heatmap')")
    reasoning: str = Field(description="Explanation of why visualization is needed or not")
    suggested_chart_title: Optional[str] = Field(description="Suggested title for the visualization if needed")

class IntentClassifier:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0)
        self.parser = PydanticOutputParser(pydantic_object=VisualizationIntent)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at analyzing user queries to determine if they need data visualization.
            Consider the following:
            1. Queries about trends over time typically need line charts
            2. Comparisons between categories often need bar charts
            3. Distribution analysis might need histograms or box plots
            4. Geographic data might need maps
            5. Simple data lookups or counts might not need visualization
            
            {format_instructions}
            """),
            ("user", "Analyze this query: {query}")
        ])

    def classify_intent(self, query: str) -> VisualizationIntent:
        """Classify if the query needs visualization and what type"""
        prompt = self.prompt.format_messages(
            query=query,
            format_instructions=self.parser.get_format_instructions()
        )
        
        response = self.llm(prompt)
        return self.parser.parse(response.content)

    def get_visualization_params(self, intent: VisualizationIntent) -> dict:
        """Get visualization parameters based on intent"""
        if not intent.needs_visualization:
            return None
            
        return {
            "type": intent.visualization_type,
            "title": intent.suggested_chart_title,
            "reasoning": intent.reasoning
        }
