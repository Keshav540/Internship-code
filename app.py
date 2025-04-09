import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

# -----------------------------
# Step 1: Scrape SHL Catalog Data
# -----------------------------
def fetch_shl_catalog():
    """
    Fetches and parses the SHL product catalog page,
    returning a DataFrame with assessment names, URLs,
    and support flags.
    """
    url = "https://www.shl.com/solutions/products/product-catalog/"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except Exception as e:
        st.error(f"Error fetching SHL catalog: {e}")
        return pd.DataFrame()

    soup = BeautifulSoup(response.text, "html.parser")
    products = []

    # Find product tiles by CSS selector (update if site changes)
    product_tiles = soup.find_all("div", class_="custom__table-responsive")
    if not product_tiles:
        product_tiles = soup.find_all("li", class_="product-item")

    for tile in product_tiles:
        # Extract name and URL
        anchor = tile.find("a", href=True)
        if anchor:
            name = anchor.get_text(strip=True)
            link = anchor["href"]
            if not link.startswith("http"):
                link = "https://www.shl.com" + link
        else:
            name, link = "Unknown", "#"

        # Extract support flags from tile text
        text = tile.get_text(" ", strip=True).lower()
        remote = "Yes" if "remote" in text else "No"
        adaptive = "Yes" if ("adaptive" in text or "irt" in text) else "No"

        products.append({
            "Assessment Name": name,
            "URL": link,
            "Remote Testing Support": remote,
            "Adaptive/IRT Support": adaptive
        })

    return pd.DataFrame(products)

# Load catalog into DataFrame
df_assessments = fetch_shl_catalog()
if df_assessments.empty:
    st.error("No product data fetched. Check website structure or network.")
else:
    st.success("Fetched SHL catalog data successfully.")

# -----------------------------
# Step 2: Extract Text from URL
# -----------------------------
def extract_text_from_url(url: str) -> str:
    """
    Fetches the given URL and concatenates all paragraph text.
    """
    try:
        resp = requests.get(url, timeout=5)
        soup = BeautifulSoup(resp.text, "html.parser")
        paras = soup.find_all("p")
        return " ".join(p.get_text() for p in paras)
    except Exception as e:
        st.error(f"Error extracting text: {e}")
        return ""

# -----------------------------
# Step 3: Recommendation Engine
# -----------------------------
def recommend_assessments(query: str, df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Builds TF-IDF vectors over assessment names only,
    computes cosine similarity with the query,
    and returns top_n recommendations.
    """
    names = df["Assessment Name"].tolist()
    corpus = [query] + names  # first element is query

    # Vectorize and compute similarity
    vec = TfidfVectorizer(stop_words="english")
    tfidf = vec.fit_transform(corpus)
    sims = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()

    # Pick top N
    idx = sims.argsort()[::-1][:top_n]
    result = df.iloc[idx].copy()
    result["Score"] = sims[idx]
    return result

# -----------------------------
# Step 4: Streamlit UI
# -----------------------------
st.title("SHL Assessment Recommendation System")

st.markdown("""
Enter a job description or URL, and get the top SHL assessment recommendations.
""")

# Choose input mode
mode = st.radio("Input type:", ["Text", "URL"])
if mode == "Text":
    user_query = st.text_area("Enter job description or query:")
else:
    url = st.text_input("Enter job description URL:")
    user_query = extract_text_from_url(url) if url else ""

if user_query:
    # Generate recommendations
    recs = recommend_assessments(user_query, df_assessments, top_n=10)

    # Prepare display table
    def linkify(row):
        return f'<a href="{row["URL"]}" target="_blank">{row["Assessment Name"]}</a>'

    display_df = recs.copy()
    display_df["Assessment Name"] = display_df.apply(linkify, axis=1)
    display_df = display_df.drop(columns=["URL", "Score"])

    st.write("### Recommendations")
    st.write(display_df.to_html(escape=False, index=False), unsafe_allow_html=True)

    # Convert to JSON and show copy/download buttons
    records = recs.drop(columns=["URL", "Score"]).to_dict(orient="records")
    json_output = json.dumps(records, indent=2)

    st.write("### JSON Output")
    st.code(json_output, language="json")

    st.download_button(
        "Download JSON",
        data=json_output,
        file_name="recommendations.json",
        mime="application/json"
    )
else:
    st.info("Please provide text or URL to get recommendations.")
    
