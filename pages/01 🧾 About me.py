import streamlit as st
from PIL import Image

st.markdown('''# About me''')

col1, col2 = st.columns([3,3 ])
img = Image.open('pages/images/logo_kleinlab_light.png')#.resize((300, 300))

with col1:
    
    st.markdown('''### 🔬 PhD Candidate at KleinLab,
                   \n  *@RC-Trust University Alliance Ruhr*
                   \n - *Bayesian uncertainty quantification for Deep Learning*
                   \n - *Quantifying predictive uncertainty in CV applications*
                   ''')
    st.write('''#''')
    st.markdown('''### 🛰️ Former ML Engineer at LiveEO
    \n **Satellite Based Infrastructure Monitoring**
                   \n - *Computer Vision applications with SAR data*''')

    st.write('''#''')

    st.markdown('''**Find the slides here:**
    \n 🔗 :blue[github.com/clarahoffmann/pycon_2023_honey_i_broke_the_pytorch_model-]''')
    st.markdown('''**and more info about me here:** 
    \n 🔗 :blue[clarahoffmann.github.io/clarahoffmann/about/]''')
    

with col2:
    st.image(img, width = 600)