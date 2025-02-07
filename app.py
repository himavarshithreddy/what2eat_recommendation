from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# Load saved data
products = joblib.load("product_data.pkl")
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
tfidf_matrix = joblib.load("tfidf_matrix.pkl")

# Define request model
class RecommendationRequest(BaseModel):
    user_scanned_product_ids: list[str]
    user_allergens: list[str]
    top_n: int = 10

@app.get("/")
def home():
    return {"message": "What2Eat Recommendation API is running!"}

@app.post("/recommend/")
def recommend(request: RecommendationRequest):
    user_scanned_product_ids = request.user_scanned_product_ids
    user_allergens = request.user_allergens
    top_n = request.top_n

    # --- FILTER OUT PRODUCTS WITH ALLERGENS ---
    def is_safe(ingredients):
        return all(allergen.lower() not in ingredients.lower() for allergen in user_allergens)
    
    filtered_df = products[products['ingredients_str'].apply(is_safe)].copy()

    # --- CATEGORY BOOSTING ---
    scanned_categories = set(
        products.loc[products['id'].isin(user_scanned_product_ids), 'categoryId']
    )
    filtered_df['category_priority'] = filtered_df['categoryId'].apply(lambda x: 1 if x in scanned_categories else 0)
    
    # --- ENSURE VALID INDICES ---
    valid_indices = filtered_df.index.intersection(range(tfidf_matrix.shape[0]))
    filtered_tfidf_matrix = tfidf_matrix[valid_indices]
    filtered_df = filtered_df.loc[valid_indices].reset_index(drop=True)

    # --- COMPUTE USER PROFILE (FIXED) ---
    scanned_indices = [i for i, product_id in enumerate(filtered_df["id"]) if product_id in user_scanned_product_ids]
    if not scanned_indices:
        return {"error": "No scanned products found in the dataset!"}
    
    user_profile = np.asarray(filtered_tfidf_matrix[scanned_indices].mean(axis=0)).reshape(1, -1)
    
    # --- COMPUTE SIMILARITY ---
    similarity_scores = cosine_similarity(user_profile, filtered_tfidf_matrix).flatten()
    filtered_df['similarity'] = similarity_scores

    # --- NORMALIZE HEALTH SCORE ---
    def normalize(series):
        return (series - series.min()) / (series.max() - series.min()) if series.max() != series.min() else series
    
    filtered_df['norm_health'] = normalize(filtered_df['healthScore'])

    # --- FINAL COMPOSITE SCORE ---
    w_similarity, w_health, w_category = 0.5, 0.2, 0.3
    filtered_df['composite_score'] = (
        w_similarity * filtered_df['similarity'] +
        w_health * filtered_df['norm_health'] +
        w_category * filtered_df['category_priority']
    )

    # --- TOP RECOMMENDATIONS ---
    recommendations = filtered_df.sort_values(by="composite_score", ascending=False).head(top_n)
    
    # --- BUILD EXPLANATIONS ---
    explanation = [
        {
            "id": row["id"],
            "name": row["name"],
            "reason": f"Similarity: {round(row['similarity'], 2)}, Health Score: {row['healthScore']}, Category match: {row['category_priority']}"
        }
        for _, row in recommendations.iterrows()
    ]

    return explanation
