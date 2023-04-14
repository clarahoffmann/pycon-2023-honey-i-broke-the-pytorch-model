import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image
from sklearn.datasets import make_circles
from streamlit_extras.badges import badge

st.markdown("# Packages for synthetic data")

tab1, tab2, tab3= st.tabs(["Tabular data", "Computer Vision", "Our status"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:

        st.subheader('timeseries-generator')
        badge(type="pypi", name="timeseries-generator", url="https://github.com/Nike-Inc/timeseries-generator")
        ts_img = Image.open('pages/images/ts_generator.png')
        st.image(ts_img)


    with col2:
        st.subheader('plaitpy')
        badge(type="pypi", name="plaitpy", url="https://github.com/plaitpy/plaitpy")
        st.write('Model fake data from yaml templates')
        st.code('''
        # a person generator
define:
  min_age: 10
  minor_age: 13
  working_age: 18

fields:
  age:
    random: gauss(25, 5)
    # minimum age is $min_age
    finalize: max($min_age, value)

  gender:
    mixture:
      - value: M
      - value: F
        ''')

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Computer Vision: ZumoLabs Zpy')
        st.write('https://github.com/ZumoLabs/zpy')
        badge(type="github", name="zpy", url="https://github.com/ZumoLabs/zpy")
        st.write('Generate synthetic data based on Blender')
        dp_gan = Image.open('pages/images/zpy_blender.png')
        st.image(dp_gan)

    with col2:
        st.subheader('Image generation based on semantic masks or layouts')
        st.write('''**Double Pooling GANs for Semantic Image Synthesis** 🔗 https://github.com/Ha0Tang/DPGAN''')
        badge(type="github", name="zpy", url="https://github.com/Ha0Tang/DPGAN")
        dp_gan = Image.open('pages/images/dp_gan_city_results.jpg')
        st.image(dp_gan)


with tab3:
  st.write('''## Let's check our debugging status...''')
  st.write('''#''')
  col1, col2, col3 = st.columns(3)

  with col1:
        st.write('''# 💾 ✅''')
        st.subheader('*:green[Data component]*')

  with col2:
        st.write('''# 🧮''')
        st.subheader('Model component')


  with col3:
        st.write('''# 🖇️''')
        st.subheader('Data & Model interplay')
