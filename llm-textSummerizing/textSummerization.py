import gradio as gr
import requests
from bs4 import BeautifulSoup
from transformers import pipeline

# Define your model name
model_name = "facebook/bart-large-cnn"

# Initialize the summarizer with the specified model
summarizer = pipeline("summarization", model=model_name)

# Function to summarize text in chunks
def summarize_text_in_chunks(text, chunk_size=1024, min_length=30):
    summaries = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size]
        max_length = max(len(chunk) // 3, min_length)
        summary = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
        summaries.append(summary)
    return " ".join(summaries)

def fetchText(url):
    # Send a GET request to the webpage
    response = requests.get(url)
    # Check if the request was successful
    if response.status_code == 200:
        print("Successfully fetched the webpage!")
    else:
        print(f"Failed to fetch the webpage. Status code: {response.status_code}")
    # Parse the HTML content of the webpage
    soup = BeautifulSoup(response.content, "html.parser")
    # Extract text from all paragraph tags
    paragraphs = soup.find_all("p")
    text = " ".join([para.get_text() for para in paragraphs])
    summary = summarize_text_in_chunks(text)
    return summary

# Set up the Gradio user interface
iface = gr.Interface(
    fn=fetchText,
    inputs=gr.Textbox(lines=2, placeholder="Enter URL here..."),
    outputs=gr.Textbox(label="Summary"),
    title="bart-large-cnn",
    description="Summarize the text using AI Model",
)

if __name__ == "__main__":
    iface.launch()
