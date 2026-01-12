import json

def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Normalize nested healthcare dataset
    specialists = []

    if isinstance(data, dict) and "doctors" in data:
        for d in data["doctors"]:
            specialists.append({
                "specialist": d.get("specialization"),
                "keywords": d.get("expertise", [])
            })
    elif isinstance(data, list):
        specialists = data
    else:
        raise ValueError("Unsupported JSON structure")

    return specialists
