import streamlit as st
import requests
import json
import pandas as pd
import uuid
import os
from datetime import datetime

# Constants
API_URL = "http://localhost:8000"
UPLOAD_ENDPOINT = f"{API_URL}/upload"
ANALYSIS_ENDPOINT = f"{API_URL}/analysis/process-data"
QUERY_ENDPOINT = f"{API_URL}/analysis/query"
CHAT_HISTORY_ENDPOINT = f"{API_URL}/analysis/chat-history"

def initialize_session_state():
    """Initialize session state variables"""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'data_processed' not in st.session_state:
        st.session_state.data_processed = False
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = "Upload"

def display_schema(schema_info):
    """Display database schema information"""
    st.subheader("Database Schema")
    
    try:
        for table_name, table_info in schema_info.items():
            st.write(f"### Table: {table_name}")
            st.write(f"Total Rows: {table_info.get('row_count', 'N/A')}")
            
            # Display columns
            st.write("#### Columns")
            columns = table_info.get('columns', [])
            for col in columns:
                col_name = col.get('name', 'Unknown')
                col_type = col.get('type', 'Unknown')
                st.text(f"{col_name} ({col_type})")
            
            # Display statistics
            st.write("#### Column Statistics")
            stats = table_info.get('statistics', {})
            for col_name, col_stats in stats.items():
                st.write(f"**{col_name}**")
                st.write(f"Type: {col_stats.get('type', 'Unknown')}")
                st.write(f"Unique Values: {col_stats.get('unique_values', 'N/A')}")
                st.write(f"Missing Values: {col_stats.get('missing_values', 'N/A')}")
                
                # Display numeric statistics if available
                if 'min' in col_stats:
                    st.write(f"Min: {col_stats['min']}")
                    st.write(f"Max: {col_stats['max']}")
                    st.write(f"Mean: {col_stats['mean']}")
                
                # Display sample values
                st.write("Sample Values:")
                sample_values = col_stats.get('sample_values', [])
                for val in sample_values:
                    st.text(f"- {val}")
                st.write("---")
    except Exception as e:
        st.error(f"Error displaying schema: {str(e)}")
        with st.expander("Debug Information"):
            st.json(schema_info)

def display_chat_interface():
    """Display chat interface"""
    st.subheader("Chat with Your Data")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "visualization" in message:
                st.image(message["visualization"])
    
    # Chat input - moved outside of any containers
    prompt = st.chat_input("Ask a question about your data")
    if prompt:
        # Add user message to chat
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get response from API
        with st.spinner("Processing your query..."):
            try:
                response = requests.post(
                    QUERY_ENDPOINT,
                    json={"prompt": prompt},
                    headers={"X-Session-ID": st.session_state.session_id}
                )
                response.raise_for_status()
                result = response.json()
                
                # Add assistant response to chat
                message = {"role": "assistant", "content": result["response"]}
                if "visualization" in result:
                    message["visualization"] = result["visualization"]
                st.session_state.chat_history.append(message)
                
                with st.chat_message("assistant"):
                    st.write(result["response"])
                    if "visualization" in result:
                        st.image(result["visualization"])
                
            except requests.exceptions.RequestException as e:
                st.error(f"Error communicating with the server: {str(e)}")
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")

def main():
    st.title("Flight Data Analysis Assistant")
    initialize_session_state()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    if st.sidebar.button("New Analysis"):
        st.session_state.data_processed = False
        st.session_state.chat_history = []
        st.session_state.current_tab = "Upload"
        st.experimental_rerun()
    
    # Main content area
    if not st.session_state.data_processed:
        st.header("Upload Data Files")
        
        # File upload
        booking_file = st.file_uploader("Upload Booking Data (CSV)", type=['csv'])
        airline_file = st.file_uploader("Upload Airline Mapping Data (CSV)", type=['csv'])
        
        # Process Data button with loading state
        process_button = st.button("Process Data", disabled=not (booking_file and airline_file))
        
        if process_button and booking_file and airline_file:
            booking_path = None
            airline_path = None
            try:
                with st.spinner("Processing your data... This may take a few minutes."):
                    # Save files temporarily
                    booking_path = f"temp_booking_{st.session_state.session_id}.csv"
                    airline_path = f"temp_airline_{st.session_state.session_id}.csv"
                    
                    # Write files to disk
                    with open(booking_path, "wb") as f:
                        f.write(booking_file.getvalue())
                    with open(airline_path, "wb") as f:
                        f.write(airline_file.getvalue())
                    
                    # Process files
                    with open(booking_path, 'rb') as booking_f, open(airline_path, 'rb') as airline_f:
                        files = {
                            'booking_file': ('booking.csv', booking_f),
                            'airline_file': ('airline.csv', airline_f)
                        }
                        
                        response = requests.post(
                            ANALYSIS_ENDPOINT,
                            files=files,
                            headers={"X-Session-ID": st.session_state.session_id}
                        )
                        response.raise_for_status()
                        result = response.json()
                    
                    # Update session state
                    st.session_state.data_processed = True
                    st.session_state.schema_info = result["schema"]
                    st.session_state.data_quality = result["data_quality"]
                    st.session_state.column_descriptions = result["column_descriptions"]
                    
                    st.success("Data processed successfully!")
                    st.experimental_rerun()
                    
            except Exception as e:
                st.error(f"Error processing data: {str(e)}")
            finally:
                # Clean up temporary files
                for path in [booking_path, airline_path]:
                    if path and os.path.exists(path):
                        try:
                            os.remove(path)
                        except Exception as e:
                            st.warning(f"Could not remove temporary file {path}: {str(e)}")
    
    else:
        # Create tabs for Schema and Chat History
        tab1, tab2 = st.tabs(["Schema", "Chat History"])
        
        with tab1:
            display_schema(st.session_state.schema_info)
        
        with tab2:
            st.subheader("Chat History")
            if not st.session_state.chat_history:
                st.info("No chat history yet. Start a conversation below!")
            else:
                # Display chat history
                for message in st.session_state.chat_history:
                    with st.chat_message(message["role"]):
                        st.write(message["content"])
                        if "visualization" in message:
                            st.image(message["visualization"])
        
        # Chat interface - placed outside of tabs
        st.write("---")
        st.subheader("Ask Questions About Your Data")
        prompt = st.chat_input("Type your question here...")
        if prompt:
            # Add user message to chat
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)
            
            # Get response from API
            with st.spinner("Processing your query..."):
                try:
                    response = requests.post(
                        f"{QUERY_ENDPOINT}?query={prompt}",  # Send query as URL parameter
                        headers={"X-Session-ID": st.session_state.session_id}
                    )
                    response.raise_for_status()
                    result = response.json()
                    
                    # Add assistant response to chat
                    message = {"role": "assistant", "content": result["response"]}
                    if "visualization" in result:
                        message["visualization"] = result["visualization"]
                    st.session_state.chat_history.append(message)
                    
                    with st.chat_message("assistant"):
                        st.write(result["response"])
                        if "visualization" in result:
                            st.image(result["visualization"])
                    
                except requests.exceptions.RequestException as e:
                    st.error(f"Error communicating with the server: {str(e)}")
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")

if __name__ == "__main__":
    main()
