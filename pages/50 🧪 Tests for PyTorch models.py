import streamlit as st
from PIL import Image
from streamlit_extras.badges import badge

st.markdown("# Tests & checks for PyTorch models")

tab1, tab2, tab3, tab4= st.tabs(["🦉 Test philosophy", "📉 torchcheck", "😋 Ingredients of a successful ML test suite", "Our status"])


with tab1:

    st.write(''' ***Classic Software Engineering testing workflows can be harmful in ML development process!***''')
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
        st.subheader('📦 torchcheck')
        st.write('''**Pre-train** checks for:
            \n - Output ranges,  invalid values, variable updates''')


        st.subheader('''**Example**:''')
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
    col1, col2 = st.columns(2)
    with col1:
        st.write('''### 💾 Tests for Dataloaders:
                    \n ***1. Data leakage***
        ''')
        st.code(''' assert not (train_set == val_set)''')
        st.write('''***2. In-batch variance:***''')
        st.code(''' it = iter(train_loader)
first = next(it)
second = next(it)
assert not (first == second)''')
        st.write('''***3. Data ranges & invalid values***''')
        st.code(''' assert not torch.sum(torch.isnan(my_tensor))).item() ''')
        st.write('''***4. Data format***''')
        st.code('''assert input.shape == (BATCH_SIZE, CHANNELS, WIDTH, HEIGHT)''')
        st.write('''**🧮 Tests for models:** Weight updates,  output ranges, output format, overfitting for trivial cases (small batches, single example) ''')
    with col2:
        st.write('''### 📁 Test structure''')
        st.write('''***Don't*** use a project dependent test structure!''')
        st.code('''
                 \n pyproject.toml
src/
    mypkg/
        train_model.py
tests/
    test_train_model.py''')

        st.write('''Set up a ***project-independent*** test suite...''')
        st.code('''
                 \n pyproject.toml
src/
    ml_test/
        dataloader_checks.py
        weight_checks.py
        directional_expectation_checks.py''')
        st.write('''... then import into ***specific ML project***''')
        st.code('''
                from ml_test.weight_checks import assert_vars_change
                ''')

with tab4:
  st.write('''## Let's check our debugging status...''')
  st.write('''#''')
  col1, col2, col3 = st.columns(3)

  with col1:
        st.write('''# 💾 ✅''')
        st.subheader('*:green[Data component]*')

  with col2:
        st.write('''# 🧮 ✅''')
        st.subheader('*:green[Model component]*')


  with col3:
        st.write('''# 🖇️''')
        st.subheader('Data & Model interplay')
