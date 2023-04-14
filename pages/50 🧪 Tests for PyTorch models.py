import streamlit as st
from PIL import Image
from streamlit_extras.badges import badge

st.markdown("# Tests & checks for PyTorch models")

tab1, tab2, tab3, tab4= st.tabs(["Test philosophy", "📉 torchcheck", "Data Leakage tests", "Trivial overfitting tests"])


with tab1:

    st.write('''**Classic Software Engineering testing workflows can be harmful in ML development process**''')
    col1, col2, col3 = st.columns([9,1,9])
    with col1:
        se_vs_ml = Image.open('pages/images/software_vs_ml_eng.png')
        st.image(se_vs_ml)
    with col3:
        ml_test_suite = Image.open('pages/images/ml_test_suite.png')
        st.image(ml_test_suite)

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Input/Output Ranges & Weight Updates')
        st.write('''Check for:
            \n - Output range of final layers, Input range of first layers, Loss functions, Invalid values

            \n - Correct weight updates ''')


        st.subheader('''**Example**: torchcheck''')
        badge(type="pypi", name="torchtest", url="https://pypi.org/project/torchtest/")
        st.code('''from torchtest import assert_vars_change ''')
        st.write('''**1. Setup data and model**''')
        st.code('''
    inputs = Variable(torch.randn(20, 20))
targets = Variable(torch.randint(0,2,(20,))).long()
batch = [inputs, targets]
model = nn.Linear(20, 2)''')
    with col2:
        st.write('''**2. Check update of parameters -> passes if nothing is frozen**''')
        st.code('''
    assert_vars_change(
        model=model,
        loss_fn=F.cross_entropy,
        optim=torch.optim.Adam(model.parameters()),
        batch=batch,
        device = 'cpu')
     ''')
        st.write('''**3. Check output ranges -> passes if output is in range (-2, 2)**''')
        st.code('''
    from torchtest import test_suite

optim = torch.optim.Adam(params_to_train)
loss_fn=F.cross_entropy

test_suite(model, loss_fn, optim, batch,
    output_range=(-2, 2),
    test_output_range=True
    )

     ''')

with tab3:
    st.header('Data Leakage tests')
    st.write('''Basic functionality to check whether variables are changing''')


with tab4:
    st.header('Data Leakage tests')
    st.write('''Basic functionality to check whether variables are changing''')
