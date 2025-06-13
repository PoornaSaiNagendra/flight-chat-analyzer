# Flight Chat Analyzer

A powerful flight data analysis system with natural language querying capabilities, built using FastAPI, Streamlit, and advanced LLM integration.

## Features

### Core Features
- **Natural Language Querying**: Ask questions about flight data in plain English
- **Advanced Data Processing**: Automated data cleaning and validation
- **Interactive Visualizations**: Dynamic charts and graphs for data insights
- **Real-time Analysis**: Instant query results with business insights
- **Multi-Model LLM Support**: OpenAI GPT-3.5 and HuggingFace Phi-3 integration with automatic fallback

### Data Analysis Capabilities
- Flight booking patterns analysis
- Airline performance metrics
- Route optimization insights
- Cancellation pattern analysis
- Seat occupancy tracking
- Delay analysis and predictions

### LLM Features
- **Dual Model Support**:
  - Primary: OpenAI GPT-3.5-turbo
  - Fallback: HuggingFace Phi-3-mini-4k-instruct
- **Automatic Fallback**: Seamless switching between models
- **Context-Aware Queries**: Schema and sample data integration
- **SQL Generation**: Natural language to SQL conversion
- **Business Insights**: Automated analysis and recommendations

## System Architecture

### Backend Components
1. **Data Processing Layer**
   - `DataProcessor`: Handles data cleaning and validation
   - `DataCleaner`: LLM-powered data cleaning
   - `DatabaseService`: Manages SQLite database operations

2. **Analysis Layer**
   - `AnalysisService`: Coordinates query processing
   - `LLMAnalyzer`: Manages LLM interactions and insights
   - `IntentClassifier`: Determines query intent and visualization needs

3. **LLM Layer**
   - `LLMRunner`: Manages multiple LLM models
   - Automatic fallback mechanism
   - Context-aware prompt management

4. **API Layer**
   - FastAPI endpoints for data processing and querying
   - WebSocket support for real-time updates
   - Structured response formatting

### Frontend Components
1. **Streamlit Interface**
   - ChatGPT-like chat interface
   - Interactive data visualizations
   - Real-time query results
   - File upload and management

2. **Visualization Components**
   - Dynamic chart generation
   - Interactive data tables
   - Business insight displays
   - Risk assessment visualization

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/flight-chat-analyzer.git
cd flight-chat-analyzer
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_openai_api_key
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
```

## Usage

1. Start the backend server:
```bash
uvicorn app.main:app --reload
```

2. Start the frontend:
```bash
streamlit run frontend/app.py
```

3. Access the application:
- Frontend: http://localhost:8501
- API Documentation: http://localhost:8000/docs

## API Endpoints

### Data Management
- `POST /analysis/process-data`: Process and validate flight data
- `GET /analysis/schema`: Get database schema
- `GET /analysis/sample-data/{table}`: Get sample data

### Query Processing
- `POST /analysis/query`: Execute natural language queries
- `GET /analysis/summary`: Get data summary statistics

## Example Queries

1. Basic Analysis:
```sql
"What are the top 3 most frequented destinations?"
"Show me the average delay time by airline"
```

2. Advanced Analysis:
```sql
"Analyze booking patterns for the last month"
"Find flights with the highest cancellation rates"
```

3. Business Insights:
```sql
"Identify routes with the highest occupancy rates"
"Show me airlines with the best on-time performance"
```

## Testing

Run the test suite:
```bash
pytest
```

Run specific test categories:
```bash
pytest tests/test_services.py  # Service tests
pytest tests/test_api.py      # API tests
pytest tests/test_frontend.py # Frontend tests
```

## Project Structure
```
flight_chat_analyzer/
├── app/                           # Backend application
│   ├── __init__.py               # Python package marker
│   ├── main.py                   # FastAPI application entry point and configuration
│   ├── routers/                  # API route definitions
│   │   └── analysis.py          # Analysis endpoints for data processing and queries
│   ├── services/                 # Core business logic services
│   │   ├── __init__.py          # Python package marker
│   │   ├── analysis_service.py   # Coordinates query processing and analysis
│   │   ├── data_processor.py     # Handles data cleaning and preprocessing
│   │   ├── db_service.py        # Database operations and SQLite management
│   │   ├── llm_analysis_service.py # LLM-powered analysis and insights generation
│   │   ├── llm_runner.py        # Manages multiple LLM models with fallback
│   │   └── sql_llm_service.py   # SQL query generation and optimization
│   └── utils/                    # Utility functions and helpers
│       ├── __init__.py          # Python package marker
│       ├── cleaning.py          # Data cleaning and validation utilities
│       ├── intent.py            # Query intent classification system
│       └── llm_analysis.py      # LLM analysis and visualization tools
├── frontend/                      # Frontend application
│   └── app.py                    # Streamlit UI with chat interface
├── tests/                         # Test suite
│   ├── __init__.py              # Python package marker
│   ├── test_analysis.py         # Tests for analysis functionality
│   ├── test_api.py              # API endpoint tests
│   ├── test_chat.py             # Chat interface and interaction tests
│   ├── test_cleaning.py         # Data cleaning utility tests
│   ├── test_frontend.py         # Frontend component tests
│   ├── test_services.py         # Service layer tests
│   └── test_upload.py           # File upload functionality tests
├── uploaded/                      # Directory for uploaded data files
├── .env                          # Environment variables configuration
├── .gitignore                    # Git ignore rules
├── README.md                     # Project documentation
└── requirements.txt              # Python dependencies
```

### Key Components

#### Backend Services
- **analysis_service.py**: Coordinates the entire analysis workflow, including query processing, data retrieval, and result formatting
- **data_processor.py**: Handles data ingestion, cleaning, and validation for both booking and airline data
- **db_service.py**: Manages SQLite database operations, including table creation, data loading, and query execution
- **llm_analysis_service.py**: Provides business insights and recommendations using LLM capabilities
- **llm_runner.py**: Manages multiple LLM models (OpenAI and HuggingFace) with automatic fallback
- **sql_llm_service.py**: Converts natural language queries to optimized SQL with validation

#### Utility Modules
- **cleaning.py**: Provides data cleaning functions and validation rules
- **intent.py**: Classifies user queries to determine appropriate analysis and visualization needs
- **llm_analysis.py**: Handles LLM-powered analysis, visualization generation, and business insights

#### Frontend
- **app.py**: Implements a ChatGPT-like interface with:
  - Real-time chat interaction
  - File upload management
  - Interactive visualizations
  - Query result display

#### Test Suite
- **test_analysis.py**: Tests analysis functionality and business insights
- **test_api.py**: Validates API endpoints and response formats
- **test_chat.py**: Tests chat interface and user interactions
- **test_cleaning.py**: Verifies data cleaning and validation
- **test_frontend.py**: Tests frontend components and UI functionality
- **test_services.py**: Validates service layer functionality
- **test_upload.py**: Tests file upload and processing

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for GPT-3.5-turbo
- HuggingFace for Phi-3 model
- FastAPI for the backend framework
- Streamlit for the frontend interface


