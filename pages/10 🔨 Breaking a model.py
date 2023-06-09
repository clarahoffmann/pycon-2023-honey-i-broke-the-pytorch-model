import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

st.title("🔨 Breaking a model")

tab1, tab2, tab3, tab4 = st.tabs(["Model components", "Breakage points", "Isolating breakage", "Debugging cookbook 😋"])
with tab1:
        col1, col2, col3 = st.columns([5,1, 6])

        with col1:
            st.subheader('Standard setup for PyTorch models')
            image_directory = "pages/images/pipeline_components.png"
            image = Image.open(image_directory)
            st.image(image, caption='Typical PyTorch model components')

        with col3:
            st.subheader('Breakage-susceptible components')
            image_directory = "pages/images/building_blocks.png"
            image = Image.open(image_directory)
            st.image(image, caption='In-detail look')

with tab2:
    col1, col2= st.columns([3,4])

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

            df_freeze_weights = pd.read_csv('pages/torch_examples/reformatted_metrics/circle_data_frozen_weights.csv')
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
            df_freeze_bias.loc[(df_freeze_bias.label == 'train_loss'),'label']='train loss (frozen bias)'
            df_freeze_bias.loc[(df_freeze_bias.label == 'val_loss'),'label']='val loss (frozen bias)'
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
            df_dataloader_broken.loc[(df_dataloader_broken.label == 'train_loss'),'label']='train loss (broken dataloader)'
            df_dataloader_broken.loc[(df_dataloader_broken.label == 'val_loss'),'label']='val loss (broken dataloader)'
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
                y='metric',
                color = 'label',
            )

        fig_metrics.update_coloraxes(showscale=False)
        fig_metrics.update_yaxes(range = [0,0.8])
        fig_metrics.update_xaxes(range = [0, 200])

    with col2:
        st.subheader('''Loss with different breakage''')
        st.plotly_chart(fig_metrics, theme="streamlit", use_container_width=True)

with tab3:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Why is debugging DL models hard?')
        st.write('''#
                    \n - ### 🐞 Bug entanglement
                    \n - ### ⏰💸 Time & cost factor
                    \n - ### 🚧 Software Engineering based tests not helpful for ML development
                    '''
                    )

    with col2:
        st.subheader('')
        image_directory = "pages/images/spiderman_bugs.jpeg"
        image = Image.open(image_directory)
        st.image(image)

with tab4:

    col1, col2, col3 = st.columns([1,1,1.2])
    with col1:
        st.write('''# \n # \n # 💾 ''')
        st.subheader('Data component')
        st.write('''### 🩹 *:green[ simple, synthetic]* \n ### *:green[ training data]*
                 ''')

    with col2:
        st.write('''# \n # \n # 🧮''')
        st.subheader('Model component')

        st.write('''### 🩹 *:green[ project-transferable]* \n ### *:green[pre-train tests]*
                 ''')

    with col3:
        st.write('''# \n # \n # 🖇️''')
        st.subheader('Data & model interplay')
        st.write('''###  🩹 *:green[ project-transferable]* \n ### *:green[post-train tests]*
                 ''')
