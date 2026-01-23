from datasets import load_dataset

# Load local files and define splits
dataset = load_dataset(
    "text",
    data_files={
        "train": "train.txt",
        "validation": "val.txt"
    }
)

# Push to Hugging Face (overwrites previous metadata)
dataset.push_to_hub("king-ki12/medical_corpus")
