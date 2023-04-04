import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go

st.title("🔨 Breaking a model")
st.sidebar.markdown("Breaking a model", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Model components", "In-depth look", "Breakage points"])
with tab1: 
        col1, col2 = st.columns([1,7])
        with col1:
            st.write(' ')

        with col2:
            image_directory = "pages/images/pipeline_components.png"
            image = Image.open(image_directory)
            st.image(image, caption='Typical PyTorch model components')

with tab2:
        image_directory = "pages/images/building_blocks.png"
        image = Image.open(image_directory)
        st.image(image, caption='In-detail look')

with tab3:
    col1, col2= st.columns([2,4])

    m = st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #dbf0db;
        color:#003300;
    }
    div.stButton > button:hover {
        background-color: #FADBD8;
        color:#ff0000;
        }
    </style>""", unsafe_allow_html=True)

    #with col4:
        #st.button('🔞 NaN inputs')
    #with col5:
        #st.button('📉 Wrong sign loss function')
    #with col6:
        #st.button('🥣 Label mixup')

    df = pd.read_csv('pages/torch_examples/reformatted_metrics/circle_data.csv')


    fig_metrics = px.line(
                df,
                x='epoch',
                y='metric', #y=['Accuracy', 'Precision', 'Recall', 'F1'],
                color = 'label',
            )

    with col1:
        st.subheader('''Model structure''')
        st.code('''self.layers = nn.Sequential(
                nn.Linear(input_dim, 10), 
                nn.ReLU(), 
                nn.Linear(10, output_dim)
        )''')

        st.subheader('''Breakage options 🔨 ''')

        if st.button('🧊 Freeze weights', key = 'freeze_weights'):

            df_freeze_weights = pd.read_csv('pages/torch_examples/reformatted_metrics/circle_data_frozen.csv')
            df_freeze_weights.loc[(df_freeze_weights.label == 'train_loss'),'label']='train loss (no relu)'
            df_freeze_weights.loc[(df_freeze_weights.label == 'val_loss'),'label']='val loss (no relu)'
            df = pd.concat([df, df_freeze_weights])

            fig_metrics = px.line(
                df,
                x='epoch',
                y='metric',
                color = 'label',
            )

        #with col2:
        if st.button('🧊 Freeze bias', key = 'freeze_bias'):
            df_freeze_bias = pd.read_csv('pages/torch_examples/reformatted_metrics/circle_data_frozen_bias.csv')
            df_freeze_bias.loc[(df_freeze_bias.label == 'train_loss'),'label']='train loss (no relu)'
            df_freeze_bias.loc[(df_freeze_bias.label == 'val_loss'),'label']='val loss (no relu)'
            df = pd.concat([df, df_freeze_bias])

            fig_metrics = px.line(
                df,
                x='epoch',
                y='metric',
                color = 'label',
            )
        
        #with col3:
        if st.button('🔄 Always return same training example'):
            df_dataloader_broken = pd.read_csv('pages/torch_examples/reformatted_metrics/circle_dataloader_broken.csv')
            df_dataloader_broken.loc[(df_dataloader_broken.label == 'train_loss'),'label']='train loss (no relu)'
            df_dataloader_broken.loc[(df_dataloader_broken.label == 'val_loss'),'label']='val loss (no relu)'
            df = pd.concat([df, df_dataloader_broken])

            fig_metrics = px.line(
                df,
                x='epoch',
                y='metric',
                color = 'label',
            )

        #with col4:
        if st.button('📈 Break activations', key = 'no_relu'):
            df_no_relu = pd.read_csv('pages/torch_examples/reformatted_metrics/circle_data_no_relu.csv')
            df_no_relu.loc[(df_no_relu.label == 'train_loss'),'label']='train loss (no relu)'
            df_no_relu.loc[(df_no_relu.label == 'val_loss'),'label']='val loss (no relu)'
            df = pd.concat([df, df_no_relu])

            fig_metrics = px.line(
                df,
                x='epoch',
                y='metric',
                color = 'label',
            )
        else:
            fig_metrics = px.line(
                df,
                x='epoch',
                y='metric', #y=['Accuracy', 'Precision', 'Recall', 'F1'],
                color = 'label',
            )

        fig_metrics.update_coloraxes(showscale=False)
        fig_metrics.update_yaxes(range = [0.3,0.8])

    with col2: 
        st.subheader('''Loss with different breakage''')
        st.plotly_chart(fig_metrics, theme="streamlit", use_container_width=True)
    

    #tab1, tab2, tab3 = st.tabs(["Loss", "Metrics", "Weight Updates"])
    #with tab1:
    #    st.write('''none''')
        #st.plotly_chart(fig_loss, theme="streamlit", use_container_width=True)
    #with tab2:
    #    st.plotly_chart(fig_metrics, theme="streamlit", use_container_width=True)
    #    fig_metrics.update_coloraxes(showscale=False)
    #with tab3:
    #    st.plotly_chart(fig_metrics, theme="streamlit", use_container_width=True)






# badges for packages: https://extras.streamlit.app/Badges

# stqdm: https://discuss.streamlit.io/t/stqdm-a-tqdm-like-progress-bar-for-streamlit/10097

# example for loss plots (scroll down): https://towardsdatascience.com/10-features-your-streamlit-ml-app-cant-do-without-implemented-f6b4f0d66d36
# https://wandb.ai/ayush-thakur/debug-neural-nets/reports/Visualizing-and-Debugging-Neural-Networks-with-PyTorch-and-Weights-Biases--Vmlldzo2OTUzNA
# https://wandb.ai/ayush-thakur/debug-neural-nets/reports/Visualizing-and-Debugging-Neural-Networks-with-PyTorch-and-Weights-Biases--Vmlldzo2OTUzNA