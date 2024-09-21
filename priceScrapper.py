import requests
from bs4 import BeautifulSoup
import json
import pandas as pd
import streamlit as st


# Function to scrape DBA search results
def scrape_dba(search_query):
    url = f"https://www.dba.dk/search/?soeg={search_query}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract product details from <script type="application/ld+json">
    product_data = []
    for script_tag in soup.find_all('script', type='application/ld+json'):
        try:
            data = json.loads(script_tag.string)
            if data["@type"] == "Product":
                product_data.append({
                    "name": data["name"],
                    "image": data["image"],
                    "url": data["url"],
                    "price": float(data["offers"]["price"]),
                    "currency": data["offers"]["priceCurrency"]
                })
        except (json.JSONDecodeError, KeyError):
            continue
    return product_data


# Search for products on DBA
search_query = st.text_input("Search for items on DBA", "Kaffekværn")
if search_query:
    st.write(f"Search results for: {search_query}")

    product_list = scrape_dba(search_query)

    if product_list:
        df = pd.DataFrame(product_list)

        # Summary statistics
        avg_price = df["price"].mean()
        min_price = df["price"].min()
        max_price = df["price"].max()

        st.write("### Price Summary")
        st.write(f"Average price: {avg_price} DKK")
        st.write(f"Minimum price: {min_price} DKK")
        st.write(f"Maximum price: {max_price} DKK")

        # Display products in grid view
        st.write("### Items found")
        num_columns = 3
        columns = st.columns(num_columns)

        for idx, product in df.iterrows():
            col = columns[idx % num_columns]
            col.image(product['image'], use_column_width=True)
            col.write(f"[{product['name']}]({product['url']})")
            col.write(f"Price: {product['price']} {product['currency']}")

    else:
        st.write("No products found.")
