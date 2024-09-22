import streamlit as st
import requests
import time

API_URL = "http://localhost:8000"  # Replace with your API URL
API_KEY = "your-api-key"  # Replace with your actual API key

st.title("RAG Pipeline Interactive Query Interface")

query = st.text_input("Enter your query:")
if st.button("Submit"):
    headers = {"X-API-Key": API_KEY}
    response = requests.post(f"{API_URL}/summarize", json={"query": query}, headers=headers)
    if response.status_code == 200:
        task_id = response.json()["task_id"]
        st.write(f"Task ID: {task_id}")
        st.write("Processing query...")
        
        # Poll for results
        while True:
            time.sleep(2)
            result_response = requests.get(f"{API_URL}/summarize/{task_id}", headers=headers)
            if result_response.status_code == 200:
                result = result_response.json()
                if result["status"] == "PENDING":
                    st.write("Still processing...")
                else:
                    st.write("Summary:")
                    st.write(result["summary"])
                    st.write("Original Query:")
                    st.write(result["original_query"])
                    st.write("Rewritten Query:")
                    st.write(result["rewritten_query"])
                    break
    else:
        st.error("An error occurred while submitting the query.")

# Add a feedback section
st.header("Feedback")
task_id_feedback = st.text_input("Task ID:")
rating = st.slider("Rating", 1, 5, 3)
comments = st.text_area("Comments:")
if st.button("Submit Feedback"):
    feedback_data = {
        "task_id": task_id_feedback,
        "rating": rating,
        "comments": comments
    }
    feedback_response = requests.post(f"{API_URL}/feedback", json=feedback_data, headers=headers)
    if feedback_response.status_code == 200:
        st.success("Feedback submitted successfully!")
    else:
        st.error("An error occurred while submitting feedback.")