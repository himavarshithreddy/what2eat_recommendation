from fastapi import FastAPI
import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# Load the trained model and data
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
tfidf_matrix = joblib.load("tfidf_matrix.pkl")
products = joblib.load("product_data.pkl")

@app.get("/")
def home():
    return {"message": "Welcome to What2Eat Recommendation API"}

@app.post("/recommend")
def recommend(user_scanned_product_ids: list, user_allergens: list, top_n: int = 10):
    """
    Get recommendations based on scanned product IDs and allergens.
    """
    
    # Filter products by allergens
    def is_safe(ingredients):
        return all(allergen.lower() not in ingredients.lower() for allergen in user_allergens)

    filtered_df = products[products['ingredients_str'].apply(is_safe)].copy()

    # Find indices of scanned products
    scanned_indices = [
        i for i, product_id in enumerate(filtered_df["id"])
        if product_id in user_scanned_product_ids
    ]

    if not scanned_indices:
        return {"message": "No scanned products found in the dataset!"}

    # Compute cosine similarity
    user_profile = tfidf_matrix[scanned_indices].mean(axis=0)
    similarity_scores = cosine_similarity(user_profile, tfidf_matrix).flatten()
    filtered_df["similarity"] = similarity_scores

    # Normalize health score
    def normalize(series):
        return (series - series.min()) / (series.max() - series.min()) if series.max() != series.min() else series

    filtered_df["norm_health"] = normalize(filtered_df["healthScore"])

    # Category priority
    category_boost = filtered_df["categoryId"].apply(lambda x: 1 if x in 
                    [prod["categoryId"] for prod in products.to_dict("records") if prod["id"] in user_scanned_product_ids] else 0)
    filtered_df["category_priority"] = category_boost

    # Compute final ranking score
    w_similarity, w_health, w_category = 0.6, 0.3, 0.1
    filtered_df["composite_score"] = (
        w_similarity * filtered_df["similarity"] +
        w_health * filtered_df["norm_health"] +
        w_category * filtered_df["category_priority"]
    )

    # Return top N recommendations
    top_recommendations = filtered_df.sort_values(by="composite_score", ascending=False).head(top_n)
    
    return top_recommendations[["id", "name", "similarity", "healthScore", "category_priority"]].to_dict(orient="records")
