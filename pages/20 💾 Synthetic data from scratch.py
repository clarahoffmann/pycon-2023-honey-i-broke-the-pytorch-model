import streamlit as st
from pages.torch_examples.data_generation import MEANS, VARIANCES, KEY, generate_mv_data, NUM_SAMPLES
import plotly.express as px
import pandas as pd
from sklearn.datasets import make_circles


st.markdown("# Synthetic data from scratch")

st.header('Synthetic data from scratch')

tab1, tab2, tab3 = st.tabs(["Linear", "Nonlinear", "Why custom data?"])
with tab1:
     col1, col2 = st.columns(2)
     with col1:
          st.write('''
               Generate from three Gaussians:
               \n $X_1 \sim \mathcal{N}((1,2)^T, \Sigma_1)$ 
               \n $X_2 \sim \mathcal{N}((-1,-2)^T, \Sigma_2)$  
               \n $X_3 \sim \mathcal{N}((0,0.5)^T, \Sigma_3)$
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

     st.write(''' Let's check what happens with this data in our model''')

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


with tab2:
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
               Somde combination of nonlinear functions will also do the job
               \n $y = \sin(x_1) + \cos(x_2) + \epsilon,$
               \n $\epsilon \sim \mathcal{N}(0,1)$
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





with tab3:
     st.write('''Data from scratch is more appropriate in research settings
              \n for example to test behavior with:''')

     lst = ['Out of distribution data', 
            'Anomaly Detection',
            'Uncertainty Estimation']

     s = ''

     for i in lst:
          s += "- " + i + "\n"

     st.markdown(s)


st.write('''Nonlinear regression model
     \n Most data you will get with this model is also available in packages - you will not really need to generate it this way
     \n only useful if you are testing specific properties of the model where you create data for certain scenarios
     \n for example out-of-distribution data, certain types of uncertainties, anomalies etc.''')

st.write('''Caution with benchmark datasets! even MNIST contains label errors''')
# https://l7.curtisnorthcutt.com/label-errors -> cleanlab   




