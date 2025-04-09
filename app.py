import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
        response.raise_for_status()  # Raise an exception for HTTP errors
    except Exception as e:
        st.error(f"Error fetching SHL catalog: {e}")
        return pd.DataFrame()

    # Parse the HTML content with BeautifulSoup
    soup = BeautifulSoup(response.text, "html.parser")
    products = []

    # Find all table row elements that might contain product information
    product_tiles = soup.find_all("tr")
    
    # Loop through each row (tile) to extract product details
    for tile in product_tiles:
        # Look for an anchor (<a>) tag with an href attribute inside the tile
        anchor = tile.find("a", href=True)
        if anchor:
            # Extract the text (assessment name) and href (URL)
            name = anchor.get_text(strip=True)
            link = anchor["href"]
            # If the link is relative, prepend the base URL
            if not link.startswith("http"):
                link = "https://www.shl.com" + link
        else:
            name, link = "Unknown", "#"

        # Extract additional information (support flags) from the text in the row
        text = tile.get_text(" ", strip=True).lower()
        remote = "Yes" if "remote" in text else "No"
        adaptive = "Yes" if ("adaptive" in text or "irt" in text) else "No"

        # Append the details to our list of products
        products.append({
            "Assessment Name": name,
            "URL": link,
            "Remote Testing Support": remote,
            "Adaptive/IRT Support": adaptive
        })

    # Convert the list of products to a pandas DataFrame and return it
    return pd.DataFrame(products)

# Load catalog data into a DataFrame
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
    # Extract the list of assessment names from the DataFrame
    names = df["Assessment Name"].tolist()

    # Fit the TF-IDF vectorizer on the assessment names only
    vec = TfidfVectorizer(stop_words="english")
    name_vectors = vec.fit_transform(names)

    # Transform the user query using the same vectorizer
    query_vector = vec.transform([query])
    
    # Compute cosine similarity between the query vector and each assessment name vector
    sims = cosine_similarity(query_vector, name_vectors).flatten()

    # Sort the indices based on descending similarity scores and select top_n recommendations
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

# Choose the input mode between manual text or a URL from which text will be extracted
mode = st.radio("Input type:", ["Text", "URL"])
if mode == "Text":
    user_query = st.text_area("Enter job description or query:")
else:
    url = st.text_input("Enter job description URL:")
    user_query = extract_text_from_url(url) if url else ""

if user_query:
    # Generate recommendations using the user's query
    recs = recommend_assessments(user_query, df_assessments, top_n=10)

    # Define a helper function to convert assessment names to clickable links
    def linkify(row):
        return f'<a href="{row["URL"]}" target="_blank">{row["Assessment Name"]}</a>'

    display_df = recs.copy()
    display_df["Assessment Name"] = display_df.apply(linkify, axis=1)
    # Remove extra columns that are not needed in the display
    display_df = display_df.drop(columns=["URL", "Score"])

    st.write("### Recommendations")
    # Display the recommendations as an HTML table
    st.write(display_df.to_html(escape=False, index=False), unsafe_allow_html=True)
else:
    st.info("Please provide text or URL to get recommendations.")
