import streamlit as st
from streamlit_extras.badges import badge

st.markdown("# Lazy synthetic data")
st.sidebar.markdown("Lazy synthetic data")

# Rules:
st.write('Useful questions:')
st.write('1. How simple can I make my data?')
st.write('2. If I wanted to sabotage my training - what would I do?')


# walkthrough example change detection


# add some visual examples here
st.write('timeseries-generator')  
badge(type="pypi", name="timeseries-generator", url="https://github.com/Nike-Inc/timeseries-generator")
st.write('faker')  
badge (type="pypi", name="faker", url="https://faker.readthedocs.io/en/master/")
badge(type="pypi", name="sklearn", url="https://scikit-learn.org/stable/datasets/sample_generators.html") #-> sklearn.datasets
badge(type="pypi", name="mesa", url="https://mesa.readthedocs.io/en/stable/")
badge(type="pypi", name="zpy", url="https://zumolabs.github.io/zpy/")
badge(type="pypi", name="plaitpy", url="https://github.com/plaitpy/plaitpy")