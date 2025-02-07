from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# Load saved model and data
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
tfidf_matrix = joblib.load("tfidf_matrix.pkl")
products = joblib.load("product_data.pkl")

@app.get("/")
def home():
    return {"message": "What2Eat Recommendation API is running!"}

@app.post("/recommend/")
def recommend(user_scanned_product_ids: list, user_allergens: list, top_n: int = 10):
    """
    Recommend products based on user's scanned products and allergens.
    """
    # Filter out products containing allergens
    def is_safe(ingredients):
        return all(allergen.lower() not in ingredients.lower() for allergen in user_allergens)

    filtered_df = products[products['ingredients_str'].apply(is_safe)].copy()
    
    # Compute similarity
    scanned_indices = [
        i for i, product_id in enumerate(filtered_df["id"])
        if product_id in user_scanned_product_ids
    ]
    
    if not scanned_indices:
        return {"error": "No scanned products found in the dataset!"}
    
    user_profile = np.asarray(tfidf_matrix[scanned_indices].mean(axis=0))
    similarity_scores = cosine_similarity(user_profile, tfidf_matrix).flatten()
    
    filtered_df["similarity"] = similarity_scores
    filtered_df["composite_score"] = 0.7 * filtered_df["similarity"] + 0.3 * filtered_df["healthScore"]
    
    recommendations = filtered_df.sort_values(by="composite_score", ascending=False).head(top_n)
    
    return recommendations[["id", "name", "similarity", "healthScore"]].to_dict(orient="records")
