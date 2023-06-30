import streamlit as st
import pandas as pd
import pickle
import webbrowser


model_min_max_scaler = pickle.load(open('models/model_min_max_scaler.pkl', 'rb'))
scaler_minmax = pickle.load(open('models/scaler_min_max.pkl', 'rb'))
scaler_min_max_y = pickle.load(open('models/scaler_min_max_y.pkl', 'rb'))


url = 'https://www.kaggle.com/code/itaygroer/diamonds-regression'

st.title('Diamond Prices Prediction App')

st.text("""
This app predicts the Diamond Prices, using machine learning with python.
numpy, pandas, seaborn, matplotlib, scikit-learn, streamlit
""")

st.text("""
On the sidebar you can select the parameters 
to predict the price of the diamond.
""")

st.text("""""")
st.text("""""")
st.text("""""")
st.text("""""")

if st.button('Source code', help='Click to open the notebook in kaggle'):
    webbrowser.open_new_tab(url)

st.sidebar.header('User Input Parameters')

def user_input_features():
    df_categorical = pd.DataFrame(columns=['cut_Ideal',
       'cut_Premium', 'cut_Very Good', 'cut_Good', 'cut_Fair', 'clarity_IF',
       'clarity_VVS1', 'clarity_VVS2', 'clarity_VS1', 'clarity_VS2',
       'clarity_SI1', 'clarity_SI2', 'clarity_I1', 'color_D', 'color_E',
       'color_F', 'color_G', 'color_H', 'color_I', 'color_J'], index=[0])
    df_categorical = df_categorical.fillna(0)
    carat = st.sidebar.slider('carat', 0.2, 5.0, 1.0)
    cut = st.sidebar.selectbox('cut', ('Fair', 'Good', 'Very Good', 'Premium', 'Ideal'))
    color = st.sidebar.selectbox('color', ('D','E','F','G','H','I','J'))
    clarity = st.sidebar.selectbox('clarity', ('I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF'))
    depth = st.sidebar.slider('depth', 43.0, 79.0, 60.0)
    table = st.sidebar.slider('table', 43.0, 95.0, 60.0)
    x = st.sidebar.slider('x', 0.001, 11.0, 5.0)
    y = st.sidebar.slider('y', 0.001, 59.0, 5.0)
    z = st.sidebar.slider('z', 0.001, 32.0, 5.0)
    data = {'carat': carat,
            'depth': depth,
            'table': table,
            'x': x,
            'y': y,
            'z': z,
            'volume': x*y*z,
            }
    df_categorical.loc[0, 'cut_'+cut] = 1
    df_categorical.loc[0, 'color_'+color] = 1
    df_categorical.loc[0, 'clarity_'+clarity] = 1
    features = pd.DataFrame(data, index=[0]).join(df_categorical)
    return features

df = user_input_features()

st.subheader('User Input parameters')

st.write(df)

st.text("""
The categorical variables are encoded using One Hot Encoding.
The numeric values need to be scaled.
""")

X = df.iloc[0, :7].values.reshape(1, -1)
X_m = scaler_minmax.transform(X)


X_m = pd.DataFrame(X_m[:7], columns=df.columns[:7]).join(df.iloc[:, 7:])

st.subheader('Data after Min Max Scaler')

st.write(X_m)

predict = st.button('Predict!')

if predict:
    st.write("## Predictions")
    st.code("""
    model.predict(df)
    """, language='python')
    st.write("### Predictions using Min Max Scaler")
    res = model_min_max_scaler.predict(X_m)
    res = scaler_min_max_y.inverse_transform(res.reshape(-1, 1))
    st.write(res)

