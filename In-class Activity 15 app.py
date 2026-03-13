import re
from urllib.parse import urlparse

import requests
import streamlit as st
from bs4 import BeautifulSoup


st.set_page_config(page_title="URL Content Extractor", page_icon="🔗")
st.title("URL Content Extractor")
st.write("Paste a URL to fetch the page text, preview the first 200 characters, and count the words.")


def normalize_url(url: str) -> str:
    url = url.strip()
    parsed = urlparse(url)
    if not parsed.scheme:
        url = f"https://{url}"
    return url


def extract_text_from_url(url: str) -> tuple[str, int]:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; URLContentExtractor/1.0)"
    }
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator=" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()
    word_count = len(text.split())
    return text, word_count


url = st.text_input("Enter a URL", placeholder="https://example.com")

if st.button("Extract Content"):
    if not url.strip():
        st.error("Please enter a URL before submitting.")
    else:
        try:
            normalized_url = normalize_url(url)
            text, word_count = extract_text_from_url(normalized_url)

            st.subheader("Preview")
            st.write(text[:200] if text else "No text content found.")

            st.subheader("Total Word Count")
            st.write(word_count)
        except requests.exceptions.RequestException:
            st.error("Failed to fetch the URL. Please check the link and try again.")
        except Exception:
            st.error("Something went wrong while processing the URL.")
