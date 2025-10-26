import requests
import streamlit

from components.config import news_api_key

@streamlit.cache_data(ttl=3600)
def get_news():
    query = (
        '"climate change" OR "global warming" OR sustainability OR '
        '"renewable energy" OR pollution OR recycling OR biodiversity OR '
        'conservation OR deforestation OR "carbon footprint"'
    )

    url = (
        "https://gnews.io/api/v4/search?"
        f"q={query}&"
        "lang=en&"
        "sortby=relevance&"
        "max=10&"
        f"apikey={news_api_key}"
    )

    response = requests.get(url)
    return(response.json())