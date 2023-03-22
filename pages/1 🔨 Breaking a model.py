import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd

st.title("🔨 Breaking a model")
st.sidebar.markdown("Breaking a model", unsafe_allow_html=True)


image_directory = "pages/images/building_blocks.png"
image = Image.open(image_directory)

st.image(image, caption='Building blocks of a PyTorch model')


st.header('Identifying point of breakage in PyTorch models')

col1, col2, col3, col4, col5, col6 = st.columns([1,1,1,1,1,1])

m = st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #F8F9F9;
    color:#212F3C;
}
div.stButton > button:hover {
    background-color: #FADBD8;
    color:#ff0000;
    }
</style>""", unsafe_allow_html=True)

with col1:
    st.button('🧊 Freeze weights')
with col2:
    st.button('🧊 Freeze bias')
with col3:
    st.button('🔄 Always return same training example')
with col4:
    st.button('🔞 NaN inputs')
with col5:
    st.button('📉 Wrong sign loss function')
with col6:
    st.button('🥣 Label mixup')

st.write('Loss plots')

x = [1,2,3,4,5,6,7,8,9,10]
y = [1,2,3,4,5,6,7,8,9,10]
df = pd.DataFrame({'epoch': x, 'loss': y, 'Accuracy': y , 'Precision': y, 'Recall': y, 'F1': y})
fig_loss = px.scatter(
    df,
    x='epoch',
    y='loss',
    size_max=60,
)

fig_metrics = px.scatter(
    df,
    x='epoch',
    y=['Accuracy', 'Precision', 'Recall', 'F1'],
    size_max=60,
)

tab1, tab2, tab3 = st.tabs(["Loss", "Metrics", "Weight Updates"])
with tab1:
    st.plotly_chart(fig_loss, theme="streamlit", use_container_width=True)
with tab2:
    st.plotly_chart(fig_metrics, theme="streamlit", use_container_width=True)
with tab3:
    st.plotly_chart(fig_metrics, theme="streamlit", use_container_width=True)


options_break = st.multiselect(
    'Choose one or more ways to break your model',
    ['Freeze weights', 
    'Freeze bias', 'Deliver training NaN data', 'Return same sample over and over again'])


options_test = st.multiselect(
    'Add one or more tests to your model',
    ['Freeze weights', 
    'Freeze bias', 'Deliver training NaN data', 'Return same sample over and over again'])

if st.button('Retrain model'):
    st.write('Why hello there')
else:
    st.write('Goodbye')


if st.button('Evaluate failure'):
    st.write('Why hello there')
else:
    st.write('Goodbye')


with st.expander("Dataloader"):
    st.markdown('<pre><code class="language-python"> import pandas as pd </code></pre>', unsafe_allow_html=True)

with st.expander("Model"):
    st.write('Goodbye')

with st.expander("Backpropagation"):
    st.write('Goodbye')




# badges for packages: https://extras.streamlit.app/Badges

# stqdm: https://discuss.streamlit.io/t/stqdm-a-tqdm-like-progress-bar-for-streamlit/10097

# example for loss plots (scroll down): https://towardsdatascience.com/10-features-your-streamlit-ml-app-cant-do-without-implemented-f6b4f0d66d36
# https://wandb.ai/ayush-thakur/debug-neural-nets/reports/Visualizing-and-Debugging-Neural-Networks-with-PyTorch-and-Weights-Biases--Vmlldzo2OTUzNA
# https://wandb.ai/ayush-thakur/debug-neural-nets/reports/Visualizing-and-Debugging-Neural-Networks-with-PyTorch-and-Weights-Biases--Vmlldzo2OTUzNA