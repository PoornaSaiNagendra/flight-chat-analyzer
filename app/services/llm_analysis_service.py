from typing import Dict, List, Optional
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI
import logging
import os
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMAnalysisService:
    def __init__(self):
        self.llm = OpenAI(temperature=0)
        self._setup_prompts()

    def _setup_prompts(self):
        """Setup all LLM prompts"""
        self.prompts = {
            'data_quality': PromptTemplate(
                input_variables=["data_summary", "inconsistencies"],
                template="""
                Analyze the following flight booking data quality issues and provide recommendations:
                
                Data Summary:
                {data_summary}
                
                Identified Inconsistencies:
                {inconsistencies}
                
                Please provide:
                1. A severity assessment for each issue
                2. Specific recommendations for data cleaning
                3. Potential impact on business analysis
                4. Suggested data validation rules
                
                Format your response as a structured JSON with the following keys:
                - severity_assessment: List of issues with severity levels
                - cleaning_recommendations: List of specific actions
                - business_impact: List of potential impacts
                - validation_rules: List of suggested rules
                """
            ),
            
            'cancellation_analysis': PromptTemplate(
                input_variables=["cancellation_data", "airline_info"],
                template="""
                Analyze the following flight cancellation patterns and provide insights:
                
                Cancellation Data:
                {cancellation_data}
                
                Airline Information:
                {airline_info}
                
                Please provide:
                1. Key patterns in cancellation behavior
                2. Potential root causes
                3. Recommendations for reducing cancellations
                4. Risk assessment for different airlines
                
                Format your response as a structured JSON with the following keys:
                - patterns: List of identified patterns
                - root_causes: List of potential causes
                - recommendations: List of actionable recommendations
                - risk_assessment: List of airline-specific risks
                """
            ),
            
            'occupancy_analysis': PromptTemplate(
                input_variables=["occupancy_data", "flight_info"],
                template="""
                Analyze the following flight occupancy patterns and provide business insights:
                
                Occupancy Data:
                {occupancy_data}
                
                Flight Information:
                {flight_info}
                
                Please provide:
                1. Key trends in seat occupancy
                2. Revenue optimization opportunities
                3. Route performance analysis
                4. Capacity planning recommendations
                
                Format your response as a structured JSON with the following keys:
                - trends: List of identified trends
                - revenue_opportunities: List of specific opportunities
                - route_analysis: List of route-specific insights
                - capacity_recommendations: List of planning recommendations
                """
            ),
            
            'delay_analysis': PromptTemplate(
                input_variables=["delay_data", "airline_info"],
                template="""
                Analyze the following flight delay patterns and provide operational insights:
                
                Delay Data:
                {delay_data}
                
                Airline Information:
                {airline_info}
                
                Please provide:
                1. Key patterns in delay behavior
                2. Operational bottlenecks
                3. Improvement recommendations
                4. Impact on customer satisfaction
                
                Format your response as a structured JSON with the following keys:
                - delay_patterns: List of identified patterns
                - bottlenecks: List of operational bottlenecks
                - improvements: List of specific recommendations
                - customer_impact: List of satisfaction impacts
                """
            )
        }

    def analyze_data_quality(self, data_summary: Dict, inconsistencies: Dict) -> Dict:
        """Analyze data quality issues using LLM"""
        try:
            chain = LLMChain(llm=self.llm, prompt=self.prompts['data_quality'])
            result = chain.run(
                data_summary=data_summary,
                inconsistencies=inconsistencies
            )
            return eval(result)  # Convert string response to dict
        except Exception as e:
            logger.error(f"Error in data quality analysis: {str(e)}")
            raise

    def analyze_cancellations(self, cancellation_data: Dict, airline_info: Dict) -> Dict:
        """Analyze cancellation patterns using LLM"""
        try:
            chain = LLMChain(llm=self.llm, prompt=self.prompts['cancellation_analysis'])
            result = chain.run(
                cancellation_data=cancellation_data,
                airline_info=airline_info
            )
            return eval(result)
        except Exception as e:
            logger.error(f"Error in cancellation analysis: {str(e)}")
            raise

    def analyze_occupancy(self, occupancy_data: Dict, flight_info: Dict) -> Dict:
        """Analyze occupancy patterns using LLM"""
        try:
            chain = LLMChain(llm=self.llm, prompt=self.prompts['occupancy_analysis'])
            result = chain.run(
                occupancy_data=occupancy_data,
                flight_info=flight_info
            )
            return eval(result)
        except Exception as e:
            logger.error(f"Error in occupancy analysis: {str(e)}")
            raise

    def analyze_delays(self, delay_data: Dict, airline_info: Dict) -> Dict:
        """Analyze delay patterns using LLM"""
        try:
            chain = LLMChain(llm=self.llm, prompt=self.prompts['delay_analysis'])
            result = chain.run(
                delay_data=delay_data,
                airline_info=airline_info
            )
            return eval(result)
        except Exception as e:
            logger.error(f"Error in delay analysis: {str(e)}")
            raise 