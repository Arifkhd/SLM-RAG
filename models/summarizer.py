from transformers import pipeline

summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn"
)

def summarize_report(text):
    if len(text.split()) < 50:
        return text

    result = summarizer(
        text,
        max_length=150,
        min_length=40,
        do_sample=False
    )
    return result[0]["summary_text"]
