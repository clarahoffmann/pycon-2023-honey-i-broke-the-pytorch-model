import streamlit as st
from PIL import Image

st.markdown("# ✔️ Robust PyTorch model setup")

col1, col2 = st.columns([1,2])

with col1: 
    st.write('''### Setup of test repository''')
    st.write('''- **Pre-train** and **post-train tests** are project transferable''')
    st.write('''-  Tests & checks should be usable **off-the-shelf** (otherwise overengineered)''')
    st.write('''#''')
    st.write('''### Advantages for development process''')
    st.write('''- Facilitate **fail fast, fail early** culture by speeding up development process''')
    st.write('''- Synthetic data helps to **disentangle** model from data bugs''')

with col2:
    img_workflow = Image.open('pages/images/tested_workflow.png')
    st.image(img_workflow)


# here example


# plot of losses

# choose certain model bugs
# then show whether they are caught by checks and tests or not (and how)


# fake tests in fixtures