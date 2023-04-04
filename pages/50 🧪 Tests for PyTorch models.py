import streamlit as st
from streamlit_extras.badges import badge

st.markdown("# Tests & checks for PyTorch models")
st.sidebar.markdown("Tests & checks for PyTorch models")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📉 Loss", "📊 Metrics", 
                                 "⚖️ Weight Updates", "📩 Dataloaders", 
                                 "⛔ Invalid value tests",
                                 "📜 Theory based checks"])


with tab1:
    st.write('timeseries-generator') 
    badge(type="pypi", name="torchcheck", url="https://github.com/pengyan510/torcheck")
    st.write('torchtest')  
    badge(type="pypi", name="torchtest", url="https://pypi.org/project/torchtest/")

    # tensorflow
    badge(type="pypi", name="torchtest", url="https://github.com/Thenerdstation/mltest")


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



