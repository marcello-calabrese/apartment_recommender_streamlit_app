# create an app that recommend seattle airbnb listings with streamlit

# import libraries
import streamlit as st
import streamlit.components.v1 as stc
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# load the dataset function
def load_data(data):
    df = pd.read_csv(data)
    return df

# create the funcion to TFIDF vectorize and cosine similarity

def vectorize_to_cosine(data):
    tfidf = TfidfVectorizer(min_df=3,  max_features=None,
            strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
            ngram_range=(1, 3), stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data)
    # get the cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

# create the recommendation function
@st.cache
def recommend_listing(name, cosine_sim, df, num_rec=5):
    # get the indices of the name
    name_indices = pd.Series(df.index, index=df['name']).drop_duplicates()
    # get the index of the name
    idx = name_indices[name]
    
    # now look the cosine similarity for that index
    sim_scores = list(enumerate(cosine_sim[idx]))
    # sort the cosine similarity
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    selected_name_indices = [i[0] for i in sim_scores[1:]]
    selected_name_scores = [i[0] for i in sim_scores[1:]]
    
    # get the dataframe and name
    result_df = df.iloc[selected_name_indices]
    result_df['score'] = selected_name_scores
    final_recommendation = result_df[['name','description', 'property_type','price']]
    return final_recommendation.head(num_rec)

### Prettify the result of the recommendation using streamlit components

RESULT_TEMP = '''
<div style="width: 100%; height: 100%; background-color: #f5f5f5">
<h4>{}</h4>

<p style="color:blue;"><span style="color:black;">Description:</span>{}</p>
<p style="color:blue;"><span style="color:black;">Space:</span>{}</p>
<p style="color:blue;"><span style="color:black;">Property Type:</span>{}</p>
<p style="color:blue;"><span style="color:black;">Price:</span>{}</p>

</div>
'''

# search for the listing
@st.cache
def search_listing_if_not_found(name, df):
    result_df = df[df['name'].str.contains(name)]
    return result_df


st.title('Seattle Airbnb Listing Recommendation')

menu = ['Home','Recommend', 'About']

choose_menu = st.sidebar.selectbox('Main Menu', menu)

# load the data
df = load_data('listings_cleaned.csv')

if choose_menu == 'Home':
    st.subheader('Home')
    st.write('This app is for recommending Airbnb listings in Seattle. You can search for a listing and get the recommendation')
    st.write('This app is created by Marcello')
    st.subheader('Dataset screenshot of first 5 rows')
    st.write(df.head(5))

elif choose_menu == 'Recommend':
    st.header('Recommend Apartments')
    cosine_sim_matrix = vectorize_to_cosine(df['name'])
    search_term = st.text_input('Search for a listing')
    num_of_rec = st.sidebar.number_input('Number of recommendations', min_value=1, max_value=10, value=5)
    if st.button('Search'):
        if search_term is not None:
            try:
                results = recommend_listing(search_term, cosine_sim_matrix, df, num_of_rec)
                with st.expander('Recommendations:'):
                    results_json = results.to_dict('index')
                    for row in results.iterrows():
                        rec_name = row[1][0]
                        rec_description = row[1][1]
                        rec_property_type = row[1][2]
                        rec_price = row[1][3]
                        st.write(f'{rec_name} - {rec_description} - {rec_property_type} - {rec_price}')
                        #stc.html(RESULT_TEMP.format(rec_name, rec_description, rec_space, rec_property_type, rec_price))
            except:
                results= 'No results found'
                st.warning(results)
                st.info('Suggested options include:')
                result_df = search_listing_if_not_found(search_term, df)
                st.dataframe(result_df.head(result_df))
                
else:
    st.subheader('About')
    st.write('This app is created by Marcello and built with Streamlit')
    st.write('The dataset is from https://www.kaggle.com/code/rdaldian/airbnb-content-based-recommendation-system/data?select=listings.csv')
    
