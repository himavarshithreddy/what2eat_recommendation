from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import firebase_admin
from firebase_admin import credentials, firestore

# Initialize FastAPI
app = FastAPI()

# Initialize Firebase Admin SDK (only once)
cred = credentials.Certificate("serviceAccountKey.json")  # Path to your service account key
firebase_admin.initialize_app(cred)
db = firestore.client()
products = None
# Load product data from Firebase at startup
def load_products_from_firebase():
    collection_ref = db.collection("products")  # Replace with your collection name
    docs = collection_ref.stream()
    product_list = [doc.to_dict() | {"id": doc.id} for doc in docs]
    
    # Convert to DataFrame
    products_df = pd.DataFrame(product_list)
    
    # Ensure required columns and types
    required_columns = {
        "id": "",
        "name": "",
        "ingredients_str": "",
        "categoryId": "",
        "pros_str": "",
        "cons_str": "",
        "healthScore": 0.0
    }
    for col, default in required_columns.items():
        if col not in products_df.columns:
            products_df[col] = default
        products_df[col] = products_df[col].fillna(default)
    
    string_columns = ["ingredients_str", "categoryId", "pros_str", "cons_str", "name"]
    for col in string_columns:
        products_df[col] = products_df[col].astype(str)
    products_df["healthScore"] = pd.to_numeric(products_df["healthScore"], errors="coerce").fillna(0.0)
    
    return products_df

# Global products variable loaded at startup
products = load_products_from_firebase()

class RecommendationRequest(BaseModel):
    user_scanned_product_ids: list[str]
    user_allergens: list[str]
    top_n: int = 10

@app.get("/")
def home():
    return {"message": "What2Eat Recommendation API is running!"}

@app.post("/recommend/")
def recommend(request: RecommendationRequest):
    # Extract request parameters
    scanned_ids = request.user_scanned_product_ids
    allergens = request.user_allergens
    top_n = request.top_n

    # 1. Filter allergens
    def is_safe(ingredients):
        return all(allergen.lower() not in ingredients.lower() for allergen in allergens)
    
    filtered_df = products[products['ingredients_str'].apply(is_safe)].copy()
    
    if filtered_df.empty:
        return {"error": "No products available after allergen filtering"}

    # 2. Create combined features
    filtered_df["combined_features"] = (
        (filtered_df["ingredients_str"] + " ") * 2 +
        filtered_df["categoryId"] + " " +
        filtered_df["pros_str"] + " " +
        filtered_df["cons_str"]
    )

    # 3. Calculate TF-IDF with bigrams
    tfidf_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    tfidf_matrix = tfidf_vectorizer.fit_transform(filtered_df["combined_features"])

    # 4. Category boosting
    scanned_categories = set(
        products.loc[products['id'].isin(scanned_ids), 'categoryId']
    )
    filtered_df['category_priority'] = filtered_df['categoryId'].apply(
        lambda x: 1 if x in scanned_categories else 0
    )

    # 5. Find scanned products in filtered dataset
    scanned_indices = [i for i, pid in enumerate(filtered_df["id"]) if pid in scanned_ids]
    if not scanned_indices:
        return {"error": "No scanned products found in filtered dataset"}

    # 6. Calculate similarities
    user_profile = np.asarray(tfidf_matrix[scanned_indices].mean(axis=0))
    similarity_scores = cosine_similarity(user_profile.reshape(1, -1), tfidf_matrix).flatten()
    filtered_df["similarity"] = similarity_scores

    # 7. Normalize health scores
    def normalize(s):
        return (s - s.min()) / (s.max() - s.min()) if s.max() != s.min() else s
    filtered_df["norm_health"] = normalize(filtered_df["healthScore"])

    # 8. Calculate composite scores
    w_similarity, w_health, w_category = 0.5, 0.2, 0.3
    filtered_df["composite_score"] = (
        w_similarity * filtered_df["similarity"] +
        w_health * filtered_df["norm_health"] +
        w_category * filtered_df["category_priority"]
    )

    # 9. Get top recommendations
    recommendations = filtered_df.sort_values("composite_score", ascending=False).head(top_n)

    # 10. Format explanations
    return [
        {
            "id": row["id"],
            "name": row["name"],
            "reason": f"Similarity: {round(row['similarity'], 2)}, "
                      f"Health Score: {row['healthScore']}, "
                      f"Category Match: {row['category_priority']}"
        }
        for _, row in recommendations.iterrows()
    ]

# Optional: Add endpoint to refresh product data
@app.post("/refresh-products/")
def refresh_products():
    global products
    products = load_products_from_firebase()
    return {"message": "Product data refreshed from Firebase"}