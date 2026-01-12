from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

def recommend_specialist(summary, specialists_json):
    summary_emb = model.encode(summary, convert_to_tensor=True)

    best_match = None
    best_score = -1

    for doc in specialists_json:
        keywords = " ".join(doc["keywords"])
        keyword_emb = model.encode(keywords, convert_to_tensor=True)

        score = util.cos_sim(summary_emb, keyword_emb).item()

        if score > best_score:
            best_score = score
            best_match = doc["specialist"]

    return best_match, round(best_score, 3)
