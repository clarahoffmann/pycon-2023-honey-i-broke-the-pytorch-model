import streamlit as st
from PIL import Image


st.markdown("# Standalone diagnostics 🩺")
st.sidebar.markdown("Standalone diagnostics 🩺")

tab1, tab2 = st.tabs(["Weight analysis", "Label analysis"])

with tab1:
    st.subheader('🎯 Predictive accuracy without training or validation data') 
    col1, col2 = st.columns(2)
    with col1:
        ww_logo = Image.open('pages/images/ww_logo.jpeg')
        st.image(ww_logo, width=200)



with tab2:
    st.subheader('🏷️ Identifying wrong labels') 
    col1, col2 = st.columns(2)
    with col1:
        cleanlab_logo = Image.open('pages/images/cleanlab_logo.png')
        st.image(cleanlab_logo, width=200)
        st.write('''Identify wrong labels in your dataset after training
        \n  - Label noise prediction
        \n  - Estimate predictive performance if labels were clean
        \n  - Estimate overall data quality''')
    with col2:
        st.write(''' code examples heres ''')
    