# Apartment_recommender_streamlit_app
Streamlit App that recommends apartmente in Seattle using the Airbnb kaggle dataset: https://www.kaggle.com/code/rdaldian/airbnb-content-based-recommendation-system/data?select=listings.csv. 
I created a streamlit app that reccomend apartments in the Airbnb listing of Seattle using the kaggle dataset (link above).

## Methodology used:
Content based using TFidVectorizer and cosine similarity from Scikit learn library. The engine recommends apartments taking in consideration the 'name' column of the dataset

## Steps to create the APP:

- Import libraries
- Load the dataset function 
- Create the function to vectorize and get the cosine similarity
- Create the recommendation function
- Create the function to search for the listng
- Create the Streamlit APP layout
- Load the data

## Sections of the APP:
- Main title
- Menu:
  - Home: with main description of the app  and a snapshot of the dataset
  - Recommend: where the recommendation happens :-)
  - About: Credits and the link to the dataset source

### _____________________________________________________________________________________________________ 
