import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import streamlit as st
import random
import os
import logging


# Attempt to download NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    nltk_available = True
except Exception as e:
    print(f"Failed to download NLTK data: {e}")
    nltk_available = False

# Set up logging
logging.basicConfig(level=logging.INFO)

@st.cache_data
def load_data():
    possible_paths = [
        '../data/rawdata.csv',
        'rawdata.csv',
        '/mount/src/industry-project/data/rawdata.csv',
        '/mount/src/industry-project/notebooks/rawdata.csv'
    ]
    
    for path in possible_paths:
        logging.info(f"Trying to load data from: {path}")
        if os.path.exists(path):
            logging.info(f"File found at: {path}")
            df = pd.read_csv(path)
            tag_columns = [f'tag_{i}' for i in range(1, 14)]
            df['text_features'] = df[tag_columns].fillna('').agg(' '.join, axis=1)
            return df
    
    logging.error("Unable to locate the data file. Please check the file location and permissions.")
    logging.info(f"Current working directory: {os.getcwd()}")
    logging.info(f"Files in current directory: {os.listdir()}")
    
    # If we can't find the file, return a dummy DataFrame
    return pd.DataFrame(columns=['product_name', 'price', 'shop_name', 'product_link', 'text_features'])

@st.cache_resource
def compute_similarity_matrix(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['text_features'])
    return cosine_similarity(tfidf_matrix, tfidf_matrix)

# Load data and compute similarity matrix
df = load_data()
if not df.empty:
    cosine_sim = compute_similarity_matrix(df)
else:
    cosine_sim = None

def summarize_product_name(name):
    if nltk_available:
        try:
            tokens = word_tokenize(name.lower())
            stop_words = set(stopwords.words('english'))
            tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
            word_freq = nltk.FreqDist(tokens)
            summary = ' '.join([word for word, _ in word_freq.most_common(2)])
            return summary.title()
        except Exception as e:
            print(f"Error in NLTK summarization: {e}")
            return fallback_summarize(name)
    else:
        return fallback_summarize(name)

def fallback_summarize(name):
    words = name.split()
    return ' '.join(words[:2]).title()

def get_synonyms(word):
    if nltk_available:
        try:
            from nltk.corpus import wordnet
            synonyms = set()
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    synonyms.add(lemma.name().lower().replace('_', ' '))
            return list(synonyms)
        except Exception as e:
            print(f"Error in getting synonyms: {e}")
            return [word]
    else:
        return [word]

def match_tags(recipient, occasion):
    recipient_synonyms = get_synonyms(recipient)
    occasion_synonyms = get_synonyms(occasion)
    matching_products = df[df['text_features'].apply(lambda x: any(tag in x.lower() for tag in recipient_synonyms + occasion_synonyms))]
    return matching_products

def get_random_from_top(products, column, ascending=False, top_percent=0.05):
    sorted_products = products.sort_values(column, ascending=ascending)
    top_n = max(1, int(len(sorted_products) * top_percent))
    return sorted_products.iloc[:top_n].sample(n=1).iloc[0]

def get_recommendations_by_category(matching_products):
    categories = {
        'Trending': lambda df: get_random_from_top(df, 'listing_age', ascending=True),
        'Most Favorited': lambda df: get_random_from_top(df, 'favorites', ascending=False),
        'Most Popular': lambda df: get_random_from_top(df, 'est_mo_sales', ascending=False),
        'Quirky': lambda df: df.sample(n=1).iloc[0],
        'Best Seller': lambda df: get_random_from_top(df, 'est_total_sales', ascending=False),
        'Highest Rated': lambda df: get_random_from_top(df, 'avg_reviews', ascending=False)
    }
    
    recommendations = {}
    for category, selection_func in categories.items():
        if not matching_products.empty:
            recommendations[category] = selection_func(matching_products)
    
    return recommendations

def get_placeholder_image(seed):
    random.seed(seed)
    color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
    return f"https://via.placeholder.com/300x300/{color[1:]}/FFFFFF?text=Etsy+Product"

def update_recommendations():
    matching_products = match_tags(st.session_state.recipient, st.session_state.occasion)
    st.session_state.recommendations = get_recommendations_by_category(matching_products)

def shuffle_product(category):
    matching_products = match_tags(st.session_state.recipient, st.session_state.occasion)
    new_recommendations = get_recommendations_by_category(matching_products)
    st.session_state.recommendations[category] = new_recommendations[category]

st.title("Etsy Gift Finder")

if df.empty:
    st.warning("The application is running with a dummy dataset due to data loading issues.")
else:
    recipient_options = ["Girlfriend", "Mom", "Dad", "Grandfather", "Grandmother", "Friend", "Coworker"]
    occasion_options = ["Birthday", "Wedding", "Valentine's Day", "Christmas", "Anniversary", "Graduation"]

    if 'recipient' not in st.session_state:
        st.session_state.recipient = recipient_options[0]
    if 'occasion' not in st.session_state:
        st.session_state.occasion = occasion_options[0]
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = None

    st.session_state.recipient = st.selectbox("Who is the gift for?", recipient_options, key='recipient_select')
    st.session_state.occasion = st.selectbox("What's the occasion?", occasion_options, key='occasion_select')

    if st.button("Surprise Me"):
        update_recommendations()

    if st.session_state.recommendations is not None:
        col1, col2, col3 = st.columns(3)
        columns = [col1, col2, col3]
        
        for i, (category, product) in enumerate(st.session_state.recommendations.items()):
            with columns[i % 3]:
                st.subheader(category)
                summarized_name = summarize_product_name(product['product_name'])
                st.write(summarized_name)
                
                placeholder_image = get_placeholder_image(product['product_name'])
                st.image(placeholder_image, use_column_width=True)
                
                st.write(f"Price: ${product['price']}")
                st.write(f"Shop: {product['shop_name']}")
                
                col_button, col_shuffle = st.columns([3, 1])
                with col_button:
                    st.markdown(f"[View on Etsy]({product['product_link']})")
                with col_shuffle:
                    st.button("🔀", key=f"shuffle_{i}", on_click=shuffle_product, args=(category,))