import streamlit as st
from streamlit_extras.badges import badge

st.markdown("# Tests for PyTorch models")
st.sidebar.markdown("Tests for PyTorch models")


st.write('timeseries-generator') 
badge(type="pypi", name="torchcheck", url="https://github.com/pengyan510/torcheck")
st.write('torchtest')  
badge(type="pypi", name="torchtest", url="https://pypi.org/project/torchtest/")

# tensorflow
badge(type="pypi", name="torchtest", url="https://github.com/Thenerdstation/mltest")


# here code snippets

# go through cases, what can we catch?
# any interesting cases that would occur by accident easily?



