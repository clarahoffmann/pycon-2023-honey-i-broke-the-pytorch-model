import streamlit as st
from PIL import Image

st.set_page_config(layout="wide")

st.markdown("""
<style>
.big-font {
    font-size:70px !important;
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>
.medium-font {
    font-size:50px !important;
}
</style>
""", unsafe_allow_html=True)


image_directory = "pages/images/pycon_pydata_logo.png"
image = Image.open(image_directory)
st.image(image)
st.markdown('<p class="big-font">Honey, I broke the PyTorch model</p>', unsafe_allow_html=True)
st.markdown('<p class="big-font">🍯😊⛏️🐍🔥🧮</p>', unsafe_allow_html=True)
st.markdown('<p class="medium-font">Clara Hoffmann - PyCon DE & PyData Berlin 2023</p>', unsafe_allow_html=True)
