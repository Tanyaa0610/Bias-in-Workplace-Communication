from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

MODEL_PATH = "models/all-MiniLM-L6-v2"

if os.path.exists(MODEL_PATH):
    model = SentenceTransformer(MODEL_PATH)
else:
    model = SentenceTransformer('all-MiniLM-L6-v2')

bias_phrases = [
    "women are emotional",
    "men are better leaders",
    "she is weak",
    "he is strong",
    "women are not technical",
    "girls are bad at math"
]

neutral_phrases = [
    "the person is capable",
    "they are skilled",
    "the individual is a good leader"
]

def semantic_bias(text):

    emb_text = model.encode([text])
    emb_bias = model.encode(bias_phrases)
    emb_neutral = model.encode(neutral_phrases)

    sim_bias = cosine_similarity(emb_text, emb_bias).mean()
    sim_neutral = cosine_similarity(emb_text, emb_neutral).mean()

    return float(sim_bias - sim_neutral)