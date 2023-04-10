import streamlit as st
from streamlit_extras.badges import badge

st.markdown("# Tests & checks for PyTorch models")
st.sidebar.markdown("Tests & checks for PyTorch models")

tab1, tab2, tab3, tab4, tab5= st.tabs(["Test philosophy", "📉 Input/Output range tests", "Data Leakage tests",  "📊 Variable change tests", "Trivial overfitting tests"])


with tab1:
    st.subheader('Test philosophy')

    st.write('''**Classic Software Enginesering testing workflows are harmful in ML development process**''')

with tab2:
    st.subheader('Input/Output range tests')
    col1, col2 = st.columns(2)
    with col1:
        st.write('''Any variables that are confined to certain ranges can be checked
            \n especially:
            \n - **Outputs of final layers**: from Sigmoid, Tanh, ReLU, Softmax
            \n - **Inputs of first layers** : Normalization ranges
            \n - **Loss functions**: in examples where the loss is expected to go in a certain direction
            \n - **Invalid values**: NaN, Inf, -Inf''')
    
    with col2:
        st.write('hello')




    
    #badge(type="pypi", name="torchcheck", url="https://github.com/pengyan510/torcheck")
    #st.write('torchtest')  
    #badge(type="pypi", name="torchtest", url="https://pypi.org/project/torchtest/")

    # tensorflow
    #badge(type="pypi", name="torchtest", url="https://github.com/Thenerdstation/mltest")


with tab3:
    st.header('torchtest')  
    st.write('''Basic functionality to check whether variables are changing''')
    
    st.subheader('''Setup variables''')
    st.code('''
    inputs = Variable(torch.randn(20, 20))
    targets = Variable(torch.randint(0,2,(20,))).long()
    batch = [inputs, targets]
    model = nn.Linear(20, 2)''')

    st.subheader('''Check update of params''')
    st.code('''
    assert_vars_change(
        model=model,
        loss_fn=F.cross_entropy,
        optim=torch.optim.Adam(model.parameters()),
        batch=batch,
        device = 'cpu')
     ''')


    badge(type="pypi", name="torchtest", url="https://pypi.org/project/torchtest/")



# here code snippets

# go through cases, what can we catch?
# any interesting cases that would occur by accident easily?



