'''
This programme creates a web app to predict Iris species based on measurements defined by the user.
'''

import streamlit as st
from calc import predict_species

# select widescreen layout and elect page title
st.set_page_config(page_title='Iris App', layout='wide')

# print title
st.title('Iris Species Predictor')

# create inputs
sepal_length = st.sidebar.slider('Sepal length:' , min_value=4.0, max_value=8.0, step=0.1)
sepal_width = st.sidebar.slider('Sepal width:' , min_value=1.8, max_value=4.5, step=0.1)
petal_length = st.sidebar.slider('Petal length:' , min_value=0.8, max_value=7.0, step=0.1)
petal_width = st.sidebar.slider('Petal width:' , min_value=0.1, max_value=2.7, step=0.1)

# calculate species
species = predict_species(sepal_length, sepal_width, petal_length, petal_width)

# print species
st.markdown(f'These measurements are typical of an *Iris {species}*.')