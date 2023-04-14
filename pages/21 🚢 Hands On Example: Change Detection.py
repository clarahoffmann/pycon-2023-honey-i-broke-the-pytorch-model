from typing import Tuple

import jax
import numpy as onp
import plotly.express as px
import streamlit as st
from jax import random
from PIL import Image

st.title('🚢 Walkthrough example: Harbor Activity')

harbor_dir = "pages/images/ship_time_series.png"
harbor_img = Image.open(harbor_dir)

tab1, tab2 = st.tabs(['Use Case', 'Simple Implementation'])
with tab1:
        col1, col2= st.columns([1,1])
        with col1:
                st.subheader('Use Case: Identify changes in harbors')
                st.image(harbor_img, caption='Changes in harbor activity over time')
                st.write('''
                \n - satellite data is expensive, often unclean (clouds, reflections)
                \n - labelling is needed at least for validation
                \n - combination of freely available datasets tedious and likely to be to heterogeneous''')

        with col2:
                st.subheader('Complicated workaround')
                expensive_pipeline_dir = "pages/images/expensive_pipeline.png"
                exp_pipeline_img = Image.open(expensive_pipeline_dir)
                st.image(exp_pipeline_img, caption='How not to create your data', width=400)
