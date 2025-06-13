import streamlit as st
import requests
import pandas as pd
import json
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

# Constants
API_BASE_URL = "http://localhost:8000"
UPLOAD_DIR = "uploaded"

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'data_processed' not in st.session_state:
    st.session_state.data_processed = False
if 'schema' not in st.session_state:
    st.session_state.schema = None
if 'sample_data' not in st.session_state:
    st.session_state.sample_data = None

def process_uploaded_files():
    """Process uploaded files and load data into the database"""
    try:
        # Get the most recent files from the upload directory
        booking_files = [f for f in os.listdir(UPLOAD_DIR) if f.endswith('.csv') and 'booking' in f.lower()]
        airline_files = [f for f in os.listdir(UPLOAD_DIR) if f.endswith('.csv') and 'airline' in f.lower()]
        
        if not booking_files or not airline_files:
            st.error("Please upload both booking and airline mapping files")
            return False
        
        # Get the most recent files
        booking_file = os.path.join(UPLOAD_DIR, sorted(booking_files)[-1])
        airline_file = os.path.join(UPLOAD_DIR, sorted(airline_files)[-1])
        
        # Process data
        response = requests.post(
            f"{API_BASE_URL}/analysis/process-data",
            params={
                "booking_file": booking_file,
                "airline_mapping_file": airline_file
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            st.session_state.schema = data['schema']
            st.session_state.sample_data = data['sample_data']
            st.session_state.data_processed = True
            return True
        else:
            st.error(f"Error processing data: {response.json()['detail']}")
            return False
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return False

def display_message(message, is_user=False):
    """Display a message in the chat interface"""
    with st.chat_message("user" if is_user else "assistant"):
        st.write(message)

def display_analysis_results(results):
    """Display analysis results in a structured format"""
    # Display SQL Query
    with st.expander("Generated SQL Query", expanded=False):
        st.code(results['query_info']['generated_sql'], language='sql')
    
    # Display Business Insights
    st.subheader("Business Insights")
    insights = results['business_insights']
    
    # Summary
    st.markdown("### Summary")
    st.write(insights['business_interpretation']['summary'])
    
    # Key Findings
    st.markdown("### Key Findings")
    for finding in insights['business_interpretation']['key_findings']:
        st.markdown(f"- {finding}")
    
    # Insights
    st.markdown("### Detailed Insights")
    for insight in insights['insights']:
        with st.expander(f"{insight['insight']} (Confidence: {insight['confidence']})"):
            st.write(f"**Significance:** {insight['significance']}")
    
    # Recommendations
    st.markdown("### Recommendations")
    for rec in insights['recommendations']:
        with st.expander(f"{rec['recommendation']} (Priority: {rec['priority']})"):
            st.write(f"**Implementation:** {rec['implementation']}")
    
    # Risks
    st.markdown("### Risks and Mitigations")
    for risk in insights['risks']:
        with st.expander(f"{risk['risk']} (Severity: {risk['severity']})"):
            st.write(f"**Mitigation:** {risk['mitigation']}")
    
    # Display Results Table
    st.subheader("Query Results")
    df = pd.DataFrame(results['results']['data'])
    st.dataframe(df)
    
    # Display metadata
    st.caption(f"Total rows: {results['results']['total_rows']}")

def main():
    st.title("Flight Data Analysis Assistant")
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("Data Upload")
        uploaded_booking = st.file_uploader("Upload Booking Data (CSV)", type=['csv'])
        uploaded_airline = st.file_uploader("Upload Airline Mapping (CSV)", type=['csv'])
        
        if uploaded_booking and uploaded_airline:
            # Save uploaded files
            os.makedirs(UPLOAD_DIR, exist_ok=True)
            booking_path = os.path.join(UPLOAD_DIR, uploaded_booking.name)
            airline_path = os.path.join(UPLOAD_DIR, uploaded_airline.name)
            
            with open(booking_path, 'wb') as f:
                f.write(uploaded_booking.getvalue())
            with open(airline_path, 'wb') as f:
                f.write(uploaded_airline.getvalue())
            
            if st.button("Process Data"):
                if process_uploaded_files():
                    st.success("Data processed successfully!")
        
        # Display schema if available
        if st.session_state.schema:
            st.header("Database Schema")
            for table, columns in st.session_state.schema.items():
                with st.expander(f"{table} Table"):
                    for col in columns:
                        st.text(f"{col['name']} ({col['type']})")
    
    # Main chat interface
    if not st.session_state.data_processed:
        st.info("Please upload and process your data files to begin analysis.")
        return
    
    # Display chat messages
    for message in st.session_state.messages:
        display_message(message['content'], message['is_user'])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your flight data..."):
        # Add user message to chat
        st.session_state.messages.append({"content": prompt, "is_user": True})
        display_message(prompt, is_user=True)
        
        # Get response from API
        try:
            response = requests.post(
                f"{API_BASE_URL}/analysis/query",
                json={"query": prompt, "max_results": 1000}
            )
            
            if response.status_code == 200:
                results = response.json()
                # Add assistant message to chat
                st.session_state.messages.append({
                    "content": "Here's the analysis of your query:",
                    "is_user": False
                })
                display_message("Here's the analysis of your query:", is_user=False)
                display_analysis_results(results)
            else:
                error_msg = f"Error: {response.json()['detail']}"
                st.session_state.messages.append({"content": error_msg, "is_user": False})
                display_message(error_msg, is_user=False)
                
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            st.session_state.messages.append({"content": error_msg, "is_user": False})
            display_message(error_msg, is_user=False)

if __name__ == "__main__":
    main()
