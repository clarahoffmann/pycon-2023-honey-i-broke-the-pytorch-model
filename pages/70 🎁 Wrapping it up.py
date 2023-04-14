import streamlit as st
from PIL import Image

st.markdown("# 🎁 Wrapping it up")

col1, col2, col3 = st.columns(3)
img_gen_img = Image.open("pages/images/battleship_sampling.png")
ml_test_suite = Image.open("pages/images/ml_test_suite.png")
cleanlab_img = Image.open("pages/images/cleanlab.png")


with col1:
    st.write('''### 1. Synthetic data''')
    st.write('''
    *How would a minimalist artist create your data?*
    ''')
    st.image(img_gen_img, width = 250)

with col2:
    st.write('''### 2. Pre-train tests & checks''')
    st.write('''*Can be as easy as one function call*''')
    st.code('''from ml_test import test_suite

model = MyFavoriteTransformer()
test_suite(model)''')

with col3:
    st.write('''### 3. Post-train checks''')
    st.write('''*Deliver metrics that are robust to data issues*''')
    st.image(cleanlab_img, width = 350)

st.write('''# \n # \n # \n # \n # \n # \n ''')

st.markdown(
        """**Check out the repository for sources and additional information:**
    \n 🔗 :blue[github.com/clarahoffmann/pycon-2023-honey-i-broke-the-pytorch-model]"""
    )
