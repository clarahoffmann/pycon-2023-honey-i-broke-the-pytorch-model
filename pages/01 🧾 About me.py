import streamlit as st
from PIL import Image

st.markdown("""# About me""")

col1, col2 = st.columns([3, 3])
img = Image.open("pages/images/logo_kleinlab_light.png")

with col1:

    st.markdown(
        """### 🔬 Ph.D. candidate at KleinLab,
                   \n  *@RC-Trust University Alliance Ruhr*
                   \n - *Bayesian uncertainty quantification for Deep Learning*
                   \n - *Quantifying predictive uncertainty in CV applications*
                   """
    )
    st.write("""#""")
    st.markdown(
        """### 🛰️ Former ML Engineer at LiveEO
    \n **Satellite Based Infrastructure Monitoring**
                   \n - *Computer Vision applications with SAR data*"""
    )

    st.write("""# \n ##""")

    st.subheader(
        """**Find the slides here:** """)
    st.subheader("""
    🔗 :blue[github.com/clarahoffmann/pycon-2023-honey-i-broke-the-pytorch-model]"""
    )



with col2:
    st.image(img, use_column_width=True)

    st.write('# \n #')
    st.subheader("""Website: 🔗 :blue[clarahoffmann.github.io/clarahoffmann/about/]"""
    )
    st.subheader(
        """**Twitter:** 🔗 :blue[ClaraHoffmann16] """
    )
