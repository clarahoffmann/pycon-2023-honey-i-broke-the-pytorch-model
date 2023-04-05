import streamlit as st
from PIL import Image

from streamlit_extras.colored_header import colored_header
from streamlit_extras.app_logo import add_logo

st.set_page_config(
    layout="wide",
)
#add_logo("pages/images/pycon_pydata_2023-min.png", height=10)


col1, col2 = st.columns([6,5])

with col1: 
    
    
    st.write('#')
    st.write('#')
    st.write('#')
    st.markdown('''# Honey, I broke the PyTorch model''')
    st.markdown('''# 🍯😊⛏️🐍🔥🧮''')

    st.write('#')

    st.markdown('''# Clara Hoffmann''')
    st.markdown('''### 17.-19.04.2023''')
    st.markdown('''### PyCon DE & PyData Berlin 2023''')

with col2:
    st.write('#')
    st.write('#')
    st.write('#')
    st.write('#')
    st.write('#')
    st.write('#')
    st.write('#')
    image = Image.open("pages/images/pycon_pydata_logo.png")
    st.image(image, width = 600)

