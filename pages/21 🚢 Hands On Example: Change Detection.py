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
        col1, col2, col3 = st.columns([1,1,1])
        with col1:
                st.subheader(''' Common data problems in development''')
                st.write('''Custom applications are often so specific that no training data is available.
                        \n - we have some data in target format available -  but only enough for validation
                        \n - we have a specific **delivery date** for data (after labeling and cleaning), but want to start development now''')
        with col2:
                st.subheader('Use Case: Identify changes in harbors')
                st.image(harbor_img, caption='Changes in harbor activity over time')
                st.write('''
                \n - satellite data is expensive, often unclean (clouds, reflections)
                \n - labelling is needed at least for validation
                \n - combination of freely available datasets tedious and likely to be to heterogeneous''')

        with col3:
                st.subheader('Complicated workaround')
                expensive_pipeline_dir = "pages/images/expensive_pipeline.png"
                exp_pipeline_img = Image.open(expensive_pipeline_dir)
                st.image(exp_pipeline_img, caption='How not to create your data', width=400)

with tab2:
        col1, col2, col3 = st.columns([2,2,2])
        with col1:

                st.subheader('Most basic example for data we need')
                st.write('''
                        \n *1. Create single images in simplest form we can with locations of objects we want to detect.*
                        \n *2. Combine timesteps to create labels we are actually interested in.*
                        \n *3. Train model and debug.*''')
                st.write('''**Our goal is to build a model and dataloaders that:**
                        \n - has a provable basic capacity to learn
                        \n - is bug free and cheap to train''')


        with col2:
                st.subheader('Image creator')
                img_gen_dir = "pages/images/battleship_sampling.png"
                img_gen_img = Image.open(img_gen_dir)
                st.image(img_gen_img, width = 400)

        with col3:
                st.subheader('Simple label creation')
                st.write('''#''')
                label_gen_dir = "pages/images/label_generation.png"
                label_gen_img = Image.open(label_gen_dir)
                st.image(label_gen_img, width = 500)
