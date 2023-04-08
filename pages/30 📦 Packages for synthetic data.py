import streamlit as st
from streamlit_extras.badges import badge

st.markdown("# Packages for synthetic data")
st.sidebar.markdown("Packages for synthetic data")

# Rules:
#st.write('Useful questions:')
#st.write('1. How simple can I make my data?')
#st.write('2. If I wanted to sabotage my training - what would I do?')


# walkthrough example change detection
tab1, tab2, tab3, tab4= st.tabs(["Timeseries-generator", "XML/Databases: faker & plaitpy", "Computer Vision: Zpy", "Agent-based: mesa"])

with tab1: 
    st.subheader('timeseries-generator')
    badge(type="pypi", name="timeseries-generator", url="https://github.com/Nike-Inc/timeseries-generator")

with tab2: 
    st.subheader('XML/Databases: faker & plaitpy')

    badge (type="pypi", name="faker", url="https://faker.readthedocs.io/en/master/")
    badge(type="pypi", name="sklearn", url="https://scikit-learn.org/stable/datasets/sample_generators.html")
    badge(type="pypi", name="plaitpy", url="https://github.com/plaitpy/plaitpy")

with tab3: 
    st.subheader('Computer Vision:: Zpy')
    badge(type="pypi", name="zpy", url="https://zumolabs.github.io/zpy/")

with tab4: 
    st.subheader('Agent-based: mesa')
    badge(type="pypi", name="mesa", url="https://mesa.readthedocs.io/en/stable/")

# add some visual examples here

