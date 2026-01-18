from transformers import pipeline
import re

# Initialize the summarizer
summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn"
)

def summarize_report(text):
    """
    Summarizes the input text and returns a point-wise summary.
    """
    # If text is very short, return as is
    if len(text.split()) < 50:
        return text

    # Generate a more detailed summary
    result = summarizer(
        text,
        max_length=300,   # increase max_length for more info
        min_length=100,   # increase min_length for more info
        do_sample=False
    )
    summary_text = result[0]["summary_text"]

    # Convert summary into bullet points
    # Split by sentences and remove empty strings
    sentences = [s.strip() for s in re.split(r'(?<=[.!?]) +', summary_text) if s.strip()]

    # Format as numbered points
    pointwise_summary = "\n".join([f"{i+1}. {s}" for i, s in enumerate(sentences)])
    
    return pointwise_summary
