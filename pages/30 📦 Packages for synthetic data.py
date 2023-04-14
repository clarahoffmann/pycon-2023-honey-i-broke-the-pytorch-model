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
        st.subheader('''sklearn sample_generators''')
        badge(type="pypi", name="sklearn", url="https://scikit-learn.org/stable/datasets/sample_generators.html")
        st.code('''
        from sklearn.datasets import make_circles
make_circles(n_samples=NUM_SAMPLES, factor=0.5, noise=0.05)''')

        data_circles, label_circles = make_circles(n_samples=500, factor=0.5, noise=0.05)
        df = pd.DataFrame({'x1': data_circles[:,0], 'x2': data_circles[:,1], 'labels': label_circles})

        fig_circles = px.scatter(
            df,
            x='x1',
            y='x2',
            color='labels',
            size_max=60,
            color_continuous_scale = px.colors.sequential.Peach,
            )
        fig_circles.update_coloraxes(showscale=False)
        st.plotly_chart(fig_circles, theme="streamlit", width = 50, use_container_width = False)

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

    with col2:
        st.subheader('Image generation based on semantic masks or layouts')
        st.write('''**Double Pooling GANs for Semantic Image Synthesis** 🔗 https://github.com/Ha0Tang/DPGAN''')
        badge(type="github", name="zpy", url="https://github.com/Ha0Tang/DPGAN")
        dp_gan = Image.open('pages/images/dp_gan_city_results.jpg')
        st.image(dp_gan)

        st.write('''**Feature Pyramid Diffusion for Complex Scene Image Synthesis** 🔗 https://github.com/davidhalladay/Frido''')
        badge(type="github", name="zpy", url="https://github.com/davidhalladay/Frido")

        frido_examples = Image.open('pages/images/frido_examples.png')
        st.image(frido_examples)


with tab3:
  st.write('''## Let's check our debugging status...''')
  st.write('''#''')
  col1, col2, col3 = st.columns(3)

  with col1:
        st.write('''# 💾 ✅''')
        st.subheader('*Data component*')

  with col2:
        st.write('''# 🧮''')
        st.subheader('Model component')


  with col3:
        st.write('''# 🖇️''')
        st.subheader('Data & Model interplay')
