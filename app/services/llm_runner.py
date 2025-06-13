# app/services/llm_runner.py
import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, SystemMessage
import logging

load_dotenv()
logger = logging.getLogger(__name__)

class LLMRunner:
    def __init__(self):
        self.openai_model = self._init_openai()
        self.huggingface_model = self._init_huggingface()
        self.current_model = self.openai_model  # Default to OpenAI

    def _init_openai(self):
        """Initialize OpenAI model"""
        try:
            return ChatOpenAI(
                temperature=0,
                model_name="gpt-3.5-turbo",
                openai_api_key=os.environ["OPENAI_API_KEY"]
            )
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI model: {str(e)}")
            return None

    def _init_huggingface(self):
        """Initialize HuggingFace model"""
        try:
            llm = HuggingFaceEndpoint(
                repo_id="microsoft/Phi-3-mini-4k-instruct",
                task="text-generation",
                max_new_tokens=512,
                do_sample=False,
                repetition_penalty=1.03,
                huggingfacehub_api_token=os.environ.get("HUGGINGFACEHUB_API_TOKEN")
            )
            return ChatHuggingFace(llm=llm)
        except Exception as e:
            logger.error(f"Failed to initialize HuggingFace model: {str(e)}")
            return None

    def _fallback_to_huggingface(self):
        """Switch to HuggingFace model if available"""
        if self.huggingface_model:
            logger.info("Falling back to HuggingFace model")
            self.current_model = self.huggingface_model
            return True
        return False

    def create_chain(self, prompt_template: str):
        """Create a LangChain chain with the given prompt template"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", prompt_template),
            ("user", "{input}")
        ])
        return LLMChain(llm=self.current_model, prompt=prompt)

    def run_query(self, query: str, schema: dict = None, sample_data: dict = None) -> str:
        """Run a query through the current model with fallback"""
        try:
            if schema and sample_data:
                prompt = """You are a data analyst. Write SQL query to answer the question.
                Use this schema for reference: {schema}
                Use this sample data for context: {sample_data}
                Question: {query}
                """
                chain = self.create_chain(prompt)
                return chain.run(query=query, schema=schema, sample_data=sample_data)
            else:
                messages = [
                    SystemMessage(content="You're a helpful assistant"),
                    HumanMessage(content=query)
                ]
                return self.current_model.invoke(messages).content
        except Exception as e:
            logger.error(f"Error with current model: {str(e)}")
            if self._fallback_to_huggingface():
                return self.run_query(query, schema, sample_data)
            raise

    def run_query_prompt(self, query: str, schema: dict, sample_data: dict) -> str:
        """Run a query through the LLM with context"""
        return self.run_query(query, schema, sample_data)

# Initialize global instance
llm_runner = LLMRunner()

# For backward compatibility
chat_model = llm_runner.current_model
