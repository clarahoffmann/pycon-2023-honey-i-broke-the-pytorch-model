import streamlit as st
from PIL import Image

st.markdown("# ✔️ Robust PyTorch model setup")

img_workflow = Image.open('pages/images/tested_workflow.png')
st.image(img_workflow)
# here example


# plot of losses

# choose certain model bugs
# then show whether they are caught by checks and tests or not (and how)


# fake tests in fixtures