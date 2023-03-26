import streamlit as st
from streamlit_extras.badges import badge


st.markdown("# Synthetic data from scratch")
st.sidebar.markdown("Synthetic data from scratch")


st.header('Synthetic data from scratch')

st.write('''Nonlinear regression model
     \n Most data you will get with this model is also available in packages - you will not really need to generate it this way
     \n only useful if you are testing specific properties of the model where you create data for certain scenarios
     \n for example out-of-distribution data, certain types of uncertainties, anomalies etc.''')


st.write('''Caution with benchmark datasets! even MNIST contains label errors''')
# https://l7.curtisnorthcutt.com/label-errors -> cleanlab   




