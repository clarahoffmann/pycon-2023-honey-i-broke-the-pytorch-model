import streamlit as st
from PIL import Image

st.markdown("# 🎁 Wrapping it up")

st.write('''### Key Takeaways''')

col1, col2, col3 = st.columns(3)
img_gen_img = Image.open("pages/images/battleship_sampling.png")
ml_test_suite = Image.open("pages/images/ml_test_suite.png")
cleanlab_img = Image.open("pages/images/cleanlab.png")


with col1:
    st.write('''**Synthetic data**''')
    st.image(img_gen_img, width = 250)
    st.write('''
    - Generating synthetic data from scratch does not have to be complicated
    ''')
    st.write('''
    - Variety of packages available to customize synthetic data with minimal effort: *timeseries-generator*, *faker*, *plaitpy*, *ZumoLabs zpy* , *mesa*
    ''')
with col2:
    st.write('''**Testing for PyTorch models**''')
    st.write('''##''')
    st.image(ml_test_suite, width = 300)
    st.write('''- *Software Engineering* based testing can be *harmful* for an ML dev process''')
    st.write('''- Experience of many ML Engineers can be *condensed* and made *transferable* to a *low-effort test suite*''')
    




with col3:
    st.write('''**Model diagnostics**''')
    st.image(cleanlab_img, width = 350)
    st.write('''- Model weights can be analyzed without data with **theoretically-founded metrics** (*weightwatcher*)
    \n - **Mislabeling of data** can be detected in an automated way (*cleanlab*)''')

st.write('''# \n 
*Additional sources for testing in ML projects:*''')
    

# illustrate pathway of debugging