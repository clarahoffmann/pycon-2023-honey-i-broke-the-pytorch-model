import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image
from sklearn.datasets import make_circles

from pages.torch_examples.data_generation import (
    KEY,
    MEANS,
    NUM_SAMPLES,
    VARIANCES,
    generate_mv_data,
)

st.markdown("# Synthetic data from scratch")

tab1, tab2, tab3 = st.tabs(["Synthetic Data", "Linear", "Nonlinear"])

with tab1:
     col1, col2 = st.columns(2)
     with col1:
          synth_vs_real_img = Image.open('pages/images/synthetic_vs_real_data.png')
          st.write(''' # ''')
          st.write(''' # ''')
          st.image(synth_vs_real_img, use_column_width=True)

     with col2:
          st.write(''' # ''')
          st.subheader('Synthetic data for debugging')
          st.write('''- We don't care about realistic looking data
                    \n - Reduce our data to the *most basic learnable characteristics*
                    \n - Want a model that is *bug-free* and *cheap to train*''')
          st.write(''' # ''')
          st.write(''' # ''')
          st.subheader('How can we create it?')
          st.write('''\n - Which *minimum requirements* must be fulfilled for our data?
                      \n - How *simple* can we make our data without losing the ability to learn?''')


with tab2:

     tab1_sub, tab2_sub = st.tabs(["Data", "Model breakage"])
     with tab1_sub:
          col1, col2 = st.columns(2)
          with col1:
               st.write('''
                    Generate from three Gaussians:
                    \n $X_1 \\sim \\mathcal{N}((1,2)^T, \\Sigma_1)$
                    \n $X_2 \\sim \\mathcal{N}((-1,-2)^T, \\Sigma_2)$
                    \n $X_3 \\sim \\mathcal{N}((0,0.5)^T, \\Sigma_3)$
                    \n
                    \n Data is lineary separable with some overlap
                    ''')

          with col2:
               normal_data, normal_labels = generate_mv_data(KEY, MEANS, VARIANCES, NUM_SAMPLES, 3)
               normal_data = normal_data.reshape(3*NUM_SAMPLES, 2)
               normal_labels = normal_labels.reshape(-1)
               df = pd.DataFrame({'x1': normal_data[:,0], 'x2': normal_data[:,1], 'labels': normal_labels})
               fig_normal = px.scatter(
                    df,
                    x='x1',
                    y='x2',
                    color='labels',
                    size_max=60,
                    color_continuous_scale = px.colors.sequential.Peach,
                    )
               fig_normal.update_coloraxes(showscale=False)
               st.plotly_chart(fig_normal, theme="streamlit", use_container_width=True)

     with tab2_sub:

          col1, col2 = st.columns(2)

          with col1:
               st.code('''
               class Encoder(nn.Module):
                    """ Simple encoder """
                    def __init__(self,
                              input_dim: int,
                              output_dim: int) -> None:
                         """Setup model layers"""
                         super().__init__()
                         self.layers = nn.Sequential(
                              nn.Linear(input_dim, 10),
                         🔨   nn.ReLU(),
                              nn.Linear(10, output_dim)
                         ) ''')

          with col2:
               df = pd.read_csv('pages/torch_examples/reformatted_metrics/linear_data.csv')

               if st.button('Break nonlinear activations', key = 'no_relu'):
                    df_no_relu = pd.read_csv('pages/torch_examples/reformatted_metrics/linear_data_no_relu.csv')
                    df_no_relu.loc[(df_no_relu.label == 'train_loss'),'label']='train loss (no relu)'
                    df_no_relu.loc[(df_no_relu.label == 'val_loss'),'label']='val loss (no relu)'

                    df_concat = pd.concat([df, df_no_relu])
                    fig_metrics = px.line(
                         df_concat,
                         x='epoch',
                         y='metric',
                         color='label',
                         )
                    fig_metrics.update_coloraxes(showscale=False)
                    st.plotly_chart(fig_metrics, theme="streamlit", use_container_width=True)
               else:
                    fig_metrics = px.line(
                         df,
                         x='epoch',
                         y='metric',
                         color='label',
                         )
                    fig_metrics.update_coloraxes(showscale=False)
                    st.plotly_chart(fig_metrics, theme="streamlit", use_container_width=True)


with tab3:

     subtab_11, subtab_12 = st.tabs(["Data", "Model breakage"])

     with subtab_11:
          col1, col2 = st.columns(2)
          with col1:
               st.write('''
                    Generate nonlinear concentric circles with
                    ''')
               st.code('''
               from sklearn.datasets import make_circles

make_circles(n_samples=NUM_SAMPLES, factor=0.5, noise=0.05)
               ''')

               st.write('''
                    Some combination of nonlinear functions will also do the job
                    \n $y = \\sin(x_1) + \\cos(x_2) + \\epsilon,$
                    \n $\\epsilon \\sim \\mathcal{N}(0,1)$
                    ''')

               data_circles, label_circles = make_circles(n_samples=NUM_SAMPLES, factor=0.5, noise=0.05)
               df = pd.DataFrame({'x1': data_circles[:,0], 'x2': data_circles[:,1], 'labels': label_circles})

          with col2:
               fig_circles = px.scatter(
                    df,
                    x='x1',
                    y='x2',
                    color='labels',
                    size_max=60,
                    color_continuous_scale = px.colors.sequential.Peach,
                    )
               fig_circles.update_coloraxes(showscale=False)
               st.plotly_chart(fig_circles, theme="streamlit", use_container_width=True)

     with subtab_12:
          col1, col2 = st.columns(2)

          with col1:
               st.code('''
               class Encoder(nn.Module):
                    """ Simple encoder """
                    def __init__(self,
                              input_dim: int,
                              output_dim: int) -> None:
                         """Setup model layers"""
                         super().__init__()
                         self.layers = nn.Sequential(
                              nn.Linear(input_dim, 10),
                         🔨   nn.ReLU(),
                              nn.Linear(10, output_dim)
                         ) ''')


          with col2:
               df = pd.read_csv('pages/torch_examples/reformatted_metrics/circle_data.csv')

               if st.button('Break nonlinear activations', key = 'no_relu_circle'):
                    df_no_relu = pd.read_csv('pages/torch_examples/reformatted_metrics/circle_data_no_relu.csv')
                    df_no_relu.loc[(df_no_relu.label == 'train_loss'),'label']='train loss (no relu)'
                    df_no_relu.loc[(df_no_relu.label == 'val_loss'),'label']='val loss (no relu)'

                    df_concat = pd.concat([df, df_no_relu])
                    fig_metrics = px.line(
                         df_concat,
                         x='epoch',
                         y='metric',
                         color='label',
                         )
                    fig_metrics.update_coloraxes(showscale=False)
                    st.plotly_chart(fig_metrics, theme="streamlit", use_container_width=True)
               else:
                    fig_metrics = px.line(
                         df,
                         x='epoch',
                         y='metric',
                         color='label',
                         )
                    fig_metrics.update_coloraxes(showscale=False)
                    st.plotly_chart(fig_metrics, theme="streamlit", use_container_width=True)
