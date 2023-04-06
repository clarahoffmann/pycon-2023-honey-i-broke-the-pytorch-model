import streamlit as st
from PIL import Image

st.markdown('''# About me''')

col1, col2 = st.columns([3,2 ])

with col1:
    st.write('''#''')
    st.markdown('''🛰️ Former ML Engineer at **LiveEO - Satellite Based Infrastructure Monitoring**
                   \n - *Computer Vision applications with SAR data*''')

    st.write('''#''')
    st.markdown('''🔬 PhD Candidate at KleinLab @**RC-Trust**,
                   *University Alliance Ruhr*
                   \n - *Bayesian uncertainty quantification for Deep Learning*
                   \n - *Quantifying predictive uncertainty in CV applications*
                   ''')

with col2:
    img = Image.open('pages/images/logo_kleinlab_light.png')#.resize((300, 300))
    st.write('''Clara Hoffmann''')
    st.image(img, width = 300)
    st.image(img, width = 250)