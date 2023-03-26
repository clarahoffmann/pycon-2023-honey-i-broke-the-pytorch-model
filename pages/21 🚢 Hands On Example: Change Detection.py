import streamlit as st
from streamlit_extras.badges import badge
from PIL import Image


st.title('🚢 Walkthrough example: Change Detection')

st.header('Use Case: Identify changes in harbor')

harbor_dir = "pages/images/ship_time_series.png"
harbor_img = Image.open(harbor_dir)
st.image(harbor_img, caption='Changes in harbor activity over time')

st.write('''Issues:
        \n - some data available but not enough to train 
        \n - how can we start building our model and make sure it can run?''')

st.write('How can we solve this problem?')

expensive_pipeline_dir = "pages/images/expensive_pipeline.png"
exp_pipeline_img = Image.open(expensive_pipeline_dir)
st.image(exp_pipeline_img, caption='How not to create your data')

# probable result: very happy prompt engineer, happy cv engineers, but result
# is more a pretrained model that was quite expensive - not our actual goal
# to have a skeleton for our model that is proven to be functionable
# What about the case in which our data is already avaiable in one or two weeks?
# we will definitely not have time to execute all above steps

# maritime traffic expert performs QA, buy software fo  r simulating ship trajectories

st.write('Does it really have to be that complicated...?')

# we want to ask ourselves: what is the simplest form to which i can reduce my
# my complex data?
# here: object there and not

st.header('Most basic example for data we need')
st.write('''Steps:
        \n 1. Create single images in simplest form we can with locations of objects we want to detect.
        \n 2. Combine timesteps to create labels we are actually interested in.
        \n 3. Train model and debug.''')

st.write('Sample coordinates -> sample random coordinates within image -> then set to 1, rest to 0')
st.write('Sample coordinates -> Can also insert landcover etc.')


st.header('Image creator')
st.write('Obtain labels by simple mathematical operations')

label_gen_dir = "pages/images/label_generation.png"
label_gen_img = Image.open(label_gen_dir)
st.image(label_gen_img, caption='Simple Ship Dataset')


# afterward train a model with this


