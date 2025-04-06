
# SHL Assessment Recommendation System

A lightweight Streamlit web app that recommends relevant SHL assessments based on a job description or natural language query.  

## Features
- Live scrape of SHL’s product catalog for up-to-date assessments  
- Content-based recommendations using TF‑IDF & cosine similarity  
- Displays assessment name, link, remote/IRT support, duration, and test time  
- Copy or download results in JSON format

## Tech Stack
- **Python 3.7+**  
- **Streamlit** for UI  
- **BeautifulSoup** & **requests** for web scraping  
- **Pandas** for data handling  
- **scikit‑learn** for TF‑IDF and similarity

## Usage
```bash
streamlit run app.py
```
1. Open the browser link.  
2. Choose “Enter Text” or “Enter URL.”  
3. Input your job description or URL.  
4. View, copy, or download the recommendations in JSON.

## Contributing
Contributions welcome! Please open issues or pull requests for bug fixes and enhancements.

## License
MIT © keshav

