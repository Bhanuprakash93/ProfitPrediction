import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objs as go
import matplotlib.pyplot as plt

# Load the saved model and scalers
best_SVR = joblib.load('best_SVR_model.pkl')
scalary = joblib.load('scalary.pkl')
scalarx = joblib.load('scalarx.pkl')

# Load the data
Corp_W_SF = pd.read_csv("C:/Users/Bhanu prakash/OneDrive - Vijaybhoomi International School/Desktop/11_07_24_Streamlit/Corp_W_SF_data.csv")

# Define functions
def fn_GRID_Data_6vars(pmin, pmax, dmMin, dmMax, GMRate, DMPromo, DMOther, PromoPenetration, OtherPenetration, EconomicIndicator, SF, bestNLM):
    CouponPenetration = np.arange(pmin, pmax, 0.005)
    DMCoupon = np.arange(dmMin, dmMax, 0.005)
    A = pd.DataFrame({'CouponPenetration': CouponPenetration})
    B = pd.DataFrame({'DMCoupon': DMCoupon})
    A['key'] = 1
    B['key'] = 1
    df = pd.merge(A, B).drop('key', axis=1)
    df['GMRate'] = GMRate
    df['DMPromo'] = DMPromo
    df['DMOther'] = DMOther
    df['PromoPenetration'] = PromoPenetration
    df['OtherPenetration'] = OtherPenetration
    df['EconomicIndicator'] = EconomicIndicator
    df['SF'] = SF
    feature_columns_NLM = ['GMRate', 'DMCoupon', 'DMPromo', 'DMOther', 'CouponPenetration', 'PromoPenetration', 'OtherPenetration', 'EconomicIndicator', 'SF']
    df_NLM = df[feature_columns_NLM]
    scaled_df_NLM = scalarx.transform(df_NLM)
    dfpreds_NLM = bestNLM.predict(scaled_df_NLM)
    dfpreds_NLM_rescaled = scalary.inverse_transform(pd.DataFrame(dfpreds_NLM))
    df['Estimated Profit'] = dfpreds_NLM_rescaled
    return df

def fn_3Dplot2(df, Exp_Month):
    fig = go.Figure(data=[go.Scatter3d(
        x=df['CouponPenetration'],
        y=df['DMCoupon'],
        z=df['Estimated Profit'],
        mode='markers',
        marker=dict(size=5, color=df['Estimated Profit'], colorscale='Viridis', opacity=0.8),
        name='Estimated_Profit'
    )])
    x1 = np.round(Exp_Month['CouponPenetration'].iloc[0], 2)
    y1 = np.round(Exp_Month['DMCoupon'].iloc[0], 2)
    z1 = np.round(Exp_Month['Avg_Profit'].iloc[0], 2)
    tit = Exp_Month['month_of_year'].iloc[0]
    fig.add_trace(go.Scatter3d(
        x=[x1], y=[y1], z=[z1],
        mode='markers',
        marker=dict(size=10, color='red', symbol='circle'),
        name=tit+"("+str(x1)+","+str(y1)+","+str(z1)+")"
    ))
    fig.update_layout(
        title='3D Plot of Estimated Profit vs CouponPenetration and DMCoupon',
        scene=dict(xaxis_title='CouponPenetration', yaxis_title='DMCoupon', zaxis_title='Estimated Profit')
    )
    st.plotly_chart(fig)

def optim_plot2D2(df, p, ML_MODEL):
    df_p = df[np.isclose(df['CouponPenetration'], p)]
    plt.figure(figsize=(10, 4))
    x = df_p['DMCoupon']
    y = df_p['Estimated Profit']
    plt.plot(x, y)
    plt.xlabel("DMCoupon")
    plt.ylabel('Estimated Profit')
    plt.title("DMCoupon vs Profit when CouponPenetration="+str(p)+"   "+ML_MODEL.__class__.__name__)
    st.pyplot(plt)

# Streamlit app
st.title('Profit Prediction Analysis')
month = st.selectbox('Select Month-Year', Corp_W_SF['month_of_year'].unique())

st.header('Input Values for Grid')
pmin = st.number_input('Enter minimum CouponPenetration (pmin) value', min_value=0.0, max_value=0.5, step=0.005)
pmax = st.number_input('Enter maximum CouponPenetration (pmax) value', min_value=0.0, max_value=0.5, step=0.005)
dmMin = st.number_input('Enter minimum DMCoupon (dmMin) value', min_value=0.0, max_value=0.9, step=0.005)
dmMax = st.number_input('Enter maximum DMCoupon (dmMax) value', min_value=0.0, max_value=0.9, step=0.005)

if st.button('Analyze'):
    Exp_Month = Corp_W_SF.loc[Corp_W_SF['month_of_year'] == month, ['GMRate', 'DMCoupon', 'DMPromo', 'DMOther', 'CouponPenetration', 'PromoPenetration', 'OtherPenetration', 'EconomicIndicator', 'SF', 'Avg_Profit', 'month_of_year']]
    Grid_df = fn_GRID_Data_6vars(pmin, pmax, dmMin, dmMax,
                                 np.round(Exp_Month['GMRate'].iloc[0], 2),
                                 np.round(Exp_Month['DMPromo'].iloc[0], 2),
                                 np.round(Exp_Month['DMOther'].iloc[0], 2),
                                 np.round(Exp_Month['PromoPenetration'].iloc[0], 2),
                                 np.round(Exp_Month['OtherPenetration'].iloc[0], 2),
                                 np.round(Exp_Month['EconomicIndicator'].iloc[0], 2),
                                 np.round(Exp_Month['SF'].iloc[0], 2),
                                 best_SVR)
    st.subheader('2D Plot')
    optim_plot2D2(Grid_df, round(Exp_Month['CouponPenetration'].iloc[0], 2), best_SVR)
    st.subheader('3D Plot')
    fn_3Dplot2(Grid_df, Exp_Month)
