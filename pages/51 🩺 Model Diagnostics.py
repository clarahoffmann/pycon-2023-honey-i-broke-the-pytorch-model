import streamlit as st
from PIL import Image
import plotly.express as px
import pandas as pd
from streamlit_extras.badges import badge

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

st.markdown("# Standalone diagnostics 🩺")
st.sidebar.markdown("Standalone diagnostics 🩺")

tab1, tab2 = st.tabs(["Weight analysis", "Label analysis"])

with tab1:
    st.subheader('🎯 Predictive accuracy without training or validation data') 
    
    col1, col2, col3 = st.columns([5,2,5])
    
    with col1: 
        st.write('''#### 📦 **WeightWatcher**''')
        #with col2:
        #    badge(type='github', name='CalculatedContent/WeightWatcher')
        #ww_logo = Image.open('pages/images/ww_logo.jpeg')
        #st.image(ww_logo, width=200)

        st.write(''' - Analyze quality of weights without training or validation data.
                    \n - Histogram of *eigenvalues* of the weight correlation matrix $X = W^TW$.
                    \n - Well trained layers should have a tail that can be described with a Power law with exponent.''')
        st.latex(r''' \alpha \approx 2 ''')
        st.write('''- High values of indicate that a layer is *not well trained*.''')
        st.write(''' *Source: Implicit Self-Regularization in Deep Neural Networks: Evidence from Random Matrix Theory and Implications for Learning, JMLR (2021)*''')

        df = pd.read_csv('pages/torch_examples/weightwatcher_metrics/circle.csv')
        df['label'] = ['no breakage']*len(df)

    with col2: 
            st.write('#')
            st.write('#')
            st.write('#')
            if st.button('Break activations'):
                df_add = pd.read_csv('pages/torch_examples/weightwatcher_metrics/circle_no_relu.csv')
                df_add['label'] = ['no relu']*len(df)
                df = pd.concat([df,df_add])
            
            if st.button('Dataloader broken'):
                df_add = pd.read_csv('pages/torch_examples/weightwatcher_metrics/circle_dataloader_broken.csv')
                df_add['label'] = ['broken dataloader']*len(df)
                df = pd.concat([df,df_add])

            if st.button('Freeze weights'):
                df_add = pd.read_csv('pages/torch_examples/weightwatcher_metrics/circle_frozen_weights.csv')
                df_add['label'] = ['frozen weights']*len(df)
                df = pd.concat([df,df_add])

            if st.button('Freeze bias'):
                df_add = pd.read_csv('pages/torch_examples/weightwatcher_metrics/circle_frozen_bias.csv')
                df_add['label'] = ['frozen bias']*len(df)
                df = pd.concat([df,df_add])
    
    with col3:
            fig_metrics = px.histogram(
                                df,
                                x='alpha',
                                nbins = 100,
                                color='label',
                                title='Alpha',
                                marginal="rug",
                                )
            fig_metrics.update_coloraxes(showscale=False)
            st.plotly_chart(fig_metrics, theme="streamlit", use_container_width=True)





with tab2:
    st.subheader('🏷️ Identifying wrong labels') 
    col1, col2 = st.columns(2)
    with col1:
        cleanlab_logo = Image.open('pages/images/cleanlab_logo.png')
        st.image(cleanlab_logo, width=200)
        st.write('''Identify wrong labels in your dataset after training
        \n  - Label noise prediction
        \n  - Estimate predictive performance if labels were clean
        \n  - Estimate overall data quality''')
    with col2:
        df = pd.read_csv('pages/torch_examples/label_mixup/circles_corrupted.csv')

        fig_circles = px.scatter(
                    df,
                    x='x1',
                    y='x2',
                    color='labels',
                    size_max=60,
                    color_continuous_scale = px.colors.sequential.Peach,
                    opacity = 0.7,
                    )

        fig_circles = fig_circles.update_coloraxes(showscale=False)
        st.plotly_chart(fig_circles, theme="streamlit", use_container_width=True)

        st.write(''' code examples heres ''')
    