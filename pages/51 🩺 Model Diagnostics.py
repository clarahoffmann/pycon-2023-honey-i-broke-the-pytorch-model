import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
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

tab1, tab2, tab3 = st.tabs(["Weight analysis", "Label analysis", "Label analysis in action"])

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
                    \n - Well trained layers should have a tail that can be described with a power law with exponent.''')
        st.latex(r''' \alpha \approx 2 ''')
        st.write('''- High values of alpha indicate that a layer is *not well trained*.''')
        st.write(''' *Source: Implicit Self-Regularization in Deep Neural Networks: Evidence from Random Matrix Theory and Implications for Learning, JMLR (2021)*''')
        st.write('''🔗 https://github.com/CalculatedContent/WeightWatcher''')

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
            fig_metrics.update_layout(xaxis_range=[0,5], yaxis_range=[0,3])
            fig_metrics.update_coloraxes(showscale=False)
            st.plotly_chart(fig_metrics, theme="streamlit", use_container_width=True)





with tab2:
    st.subheader('🏷️ Identifying wrong labels')
    col1, col2 = st.columns(2)
    with col1:
        #cleanlab_logo = Image.open('pages/images/cleanlab_logo.png')
        #st.image(cleanlab_logo, width=200)
        st.subheader('''Cleanlab''')

        st.write('''Identify *curable* data issues based on a *trained* classifier
                    \n - *Mislabeled examples* based on class scores
                    \n - Consensus with *multiple annotators*
                    \n - Scores to guide *active learning*
                    \n - Outlier/Out-of-distribution detection''')
        self_confidence_img = Image.open('pages/images/self_confidence.png')
        st.image(self_confidence_img, width = 200)
        st.write('''🔗 https://github.com/cleanlab/cleanlab''')

    with col2:
        st.subheader('''Practical Example''')
        st.write('**1. Reformat *trained* PyTorch model to sklearn object**')
        st.code('''
        model_skorch = NeuralNetClassifier(simple_dnn.encoder)
cl = cleanlab.classification.CleanLearning(model_skorch)
    ''')
        st.write('**2. Prediction with cross validation**')
        st.code('''
                   pred_probs = cross_val_predict(
                        model_skorch,
                        data_circles,
                        label_circles,
                        cv=3,
                        method="predict_proba",
                    )
predicted_labels = pred_probs.argmax(axis=1)''')
        st.write('**3. Rank label issues**')
        st.code('''
                  ranked_label_issues = find_label_issues(
                  label_circles,
                  pred_probs,
                  return_indices_ranked_by="self_confidence",
            )''')


with tab3:
    st.subheader('Label Analysis in Action')

    df = pd.read_csv('pages/torch_examples/metrics_csv/circles_uncorrupted.csv')

    cleanlab = False
    col1, col2, col3 = st.columns([1,2,5])
    with col1:
        if st.button('''Mix labels 🎲'''):
            df = pd.read_csv('pages/torch_examples/metrics_csv/circles_corrupted_test.csv')
    with col2:
        if st.button('''🫧 Cleanlab prediction'''):
            cleanlab = True
            df = pd.read_csv('pages/torch_examples/metrics_csv/circles_corrupted_test.csv')
            df_corrupted_pred = pd.read_csv('pages/torch_examples/metrics_csv/circles_corrupted_cleanlab_pred.csv')

    col1, col2 = st.columns(2)
    with col1:
        fig_circles = px.scatter(
                    df,
                    x='x1',
                    y='x2',
                    color='labels',
                    size_max=60,
                    color_continuous_scale = px.colors.sequential.Bluered,
                    opacity = 0.3,
                    title='(Un-)Corrupted Data'
                    )

        if cleanlab:
            fig_circles.add_trace(go.Scatter(
                x=df_corrupted_pred['x1'],
                y=df_corrupted_pred['x2'],
                mode='markers',
                marker=dict(
                    size=15,
                    color='green',
                    opacity=0.2
                ),
                name='cleanlab prediction'
            ))

        fig_circles = fig_circles.update_coloraxes(showscale=False)
        fig_circles.update_layout(legend=dict(
                                    yanchor="top",
                                    y=0.99,
                                    xanchor="left",
                                    x=0.01
                                ))
        st.plotly_chart(fig_circles, theme="streamlit", use_container_width=True)

    with col2:
        df = pd.read_csv('pages/torch_examples/reformatted_metrics/circles_cleanlab.csv')
        df_corrupted = pd.read_csv('pages/torch_examples/reformatted_metrics/cleanlab_data.csv') # circles_cleanlab_corrupted

        df_corrupted.loc[(df_corrupted.label == 'train_loss'),'label']='train loss (label mixup)'
        df_corrupted.loc[(df_corrupted.label == 'val_loss'),'label']='val loss (label mixup)'

        df_plot = pd.concat([df, df_corrupted])
        fig_loss = px.line(
                df_plot,
                x='epoch',
                y='metric',
                color='label',
                title='Loss',
                )

        fig_loss.update_coloraxes(showscale=False)
        st.plotly_chart(fig_loss, theme="streamlit", use_container_width=True)
