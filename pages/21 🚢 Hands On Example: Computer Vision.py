from typing import Tuple

import jax
import numpy as onp
import plotly.express as px
import streamlit as st
from jax import random
from PIL import Image

KEY = random.PRNGKey(42)

def generate_rectangle_coordinates(rng_key: random.PRNGKey, img_width: int, img_height: int, ship_width: Tuple, ship_height: Tuple) -> Tuple:
        _, key_1, key_2 = jax.random.split(rng_key, 3)
        # sample start coordinates
        x = random.randint(key_1,(1,),  0, img_width - ship_width[1])
        y = random.randint(key_2,(1,), 0, img_height - ship_height[1])

        random_width = random.randint(key_1,(1,),  ship_width[0],  ship_width[1])
        random_height = random.randint(key_2,(1,),  ship_height[0],  ship_height[1])
        return x, y, x + random_width, y + random_height

def make_image(width: int, height: int, ship_width: Tuple, ship_height: Tuple, num_ships: int) -> onp.array:
        img = onp.zeros((width, height))
        for i in range(num_ships):
                KEY = random.PRNGKey(i)
                ship_coords = generate_rectangle_coordinates(KEY, width, height, ship_width, ship_height)
                img[int(ship_coords[0]):int(ship_coords[2]), int(ship_coords[1]):int(ship_coords[3])] = 1
        return img

st.title('🚢 Walkthrough example: Harbor Activity')

harbor_dir = "pages/images/ship_time_series.png"
harbor_img = Image.open(harbor_dir)

tab1, tab2 = st.tabs(['Use Case', 'Simple Implementation'])
with tab1:
        st.subheader('Example case: Identify changes in harbor activity over time')
        st.image(harbor_img, caption='Changes in harbor activity over time')
        st.write('''
        \n #### ⏱️ Too many timesteps
        \n #### 💰 Data too expensive
        \n #### 📩 Data will arrive at some point, but not sure of quality''')


with tab2:
        col1, col2 = st.columns([1,2])

        with col1:
                st.subheader('Dataset generation...')
                num_ships = st.slider(
                'Select number of ships',
                1, 20, 1, key = 'number of ships slider')

                ship_width_input = st.slider(
                'Select min and max ship width',
                1, 512, (25, 75), key = 'width slider')

                ship_height_input = st.slider(
                'Select min and max ship height',
                1, 512, (25, 75), key = 'height slider')


                img = make_image(512, 512, (ship_width_input[0], ship_width_input[1]), (ship_height_input[0], ship_height_input[1]), num_ships)


                fig = px.imshow(img)
                fig.update_coloraxes(showscale=False)
                fig.update_layout(showlegend=False)
                fig.update_xaxes(visible=False)
                fig.update_yaxes(visible=False)

                st.plotly_chart(fig, theme="streamlit", use_container_width=True)


        with col2:
                st.subheader('... with two functions only')
                st.code('''
                        def generate_rectangle_coordinates(rng_key: random.PRNGKey, img_width: int,
                                                        img_height: int, ship_width: Tuple,
                                                        ship_height: Tuple) -> Tuple:
                                _, key_1, key_2 = jax.random.split(rng_key, 3)
                                # sample origin coordinates of ship
                                x = random.randint(key_1,(1,),  0, img_width - ship_width[1])
                                y = random.randint(key_2,(1,), 0, img_height - ship_height[1])

                                # random width and height of ship
                                random_width = random.randint(key_1,(1,),  ship_width[0],  ship_width[1])
                                random_height = random.randint(key_2,(1,),  ship_height[0],  ship_height[1])
                                return x, y, x + random_width, y + random_height
                        ''', language="python")

                st.code('''
                        def make_image(width: int, height: int, ship_width: Tuple,
                                ship_height: Tuple, num_ships: int) -> onp.array:
                                img = onp.zeros((width, height))
                                # generate ships
                                for i in range(num_ships):
                                        KEY = random.PRNGKey(i)
                                        ship_coords = generate_rectangle_coordinates(KEY, width,
                                                                height, ship_width, ship_height)
                                        img[int(ship_coords[0]):int(ship_coords[2]),
                                        int(ship_coords[1]):int(ship_coords[3])] = 1
                                return img
                        ''', language="python")
