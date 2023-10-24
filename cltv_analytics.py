import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import time
import plotly.express as px

import warnings
warnings.filterwarnings('ignore')



github_csv_url = "https://raw.githubusercontent.com/JuanCVanegas/Customer-Acquisition-Data/main/customer_acquisition_data.csv"
data = pd.read_csv(github_csv_url)
data.columns = data.columns.str.capitalize()

st.set_page_config(
    page_title = 'Customer Lifetime Value Analytics',
    page_icon = '📊',
    layout = 'wide'
)

st.title(':blue[_Customer Lifetime Value Analytics_] 🖥️')

type_of_analysis = st.selectbox(":blue[**Type of Analysis** 💡]", ("Customer Acquisition Cost","Conversion Rate","ROI", "Customer Lifetime Value"))
placeholder = st.empty()
channel_groups = data.groupby('Channel')
data['Cost'] = np.random.uniform(10, 10.7, len(data))
data['Conversion_rate'] = np.random.uniform(0.095, 0.11, len(data))
data['Revenue'] = np.random.uniform(17, 19, len(data))


for seconds in range(300):
    # Generate random values for cost, conversion_rate, and revenue for each channel
    random_cost_changes = (np.random.choice(range(-8,8), len(data['Channel'])) / 100)
    random_conversion_rate_changes = (np.random.choice(range(-50, 52), len(data['Channel'])) / 100)
    random_revenue_changes = (np.random.choice(range(-11, 11), len(data['Channel'])) / 100)

    # Update the data using vectorized operations
    data['Cost'] = data['Cost']*(1 + random_cost_changes)
    data['Conversion_rate'] = data['Conversion_rate']*(1 + random_conversion_rate_changes)
    data['Revenue'] = data['Revenue']*(1 + random_revenue_changes)

    # Recalculate ROI and CLTV for each row after the changes
    data['ROI'] = (data['Revenue'] - data['Cost']) / data['Cost']
    data['CLTV'] = (data['Revenue'] - data['Cost']) 

    if type_of_analysis == "Customer Acquisition Cost" :
        cost_by_channel = channel_groups['Cost'].mean().reset_index()
        sorted_costs = cost_by_channel.sort_values(by='Cost', ascending=False)
        first_highest = f"{sorted_costs['Channel'].iloc[0].title()}: $ {sorted_costs['Cost'].iloc[0]:.2f}"
        second_highest = f"{sorted_costs['Channel'].iloc[1].title()}: $ {sorted_costs['Cost'].iloc[1]:.2f}"
        third_highest = f"{sorted_costs['Channel'].iloc[2].title()}: $ {sorted_costs['Cost'].iloc[2]:.2f}"
        fig = px.bar(cost_by_channel, x='Channel', y='Cost', color='Channel', labels={'Cost': 'Customer Acquisition Cost'}, title='Customer Acquisition Cost by Channel')
    elif type_of_analysis == "Conversion Rate" :
        conversion_rate_by_channel = channel_groups['Conversion_rate'].mean().reset_index()
        sorted_conversation_rates = conversion_rate_by_channel.sort_values(by='Conversion_rate', ascending=False)
        first_highest = f"{sorted_conversation_rates['Channel'].iloc[0].title()}: {sorted_conversation_rates['Conversion_rate'].iloc[0]*100:.2f}%"
        second_highest = f"{sorted_conversation_rates['Channel'].iloc[1].title()}: {sorted_conversation_rates['Conversion_rate'].iloc[1]*100:.2f}%"
        third_highest = f"{sorted_conversation_rates['Channel'].iloc[2].title()}: {sorted_conversation_rates['Conversion_rate'].iloc[2]*100:.2f}%"
        fig = px.bar(conversion_rate_by_channel, x='Channel', y='Conversion_rate', color='Channel', title='Conversion Rate by Channel')
    elif type_of_analysis == "ROI" :
        roi_by_channel = data.groupby('Channel')['ROI'].mean().reset_index()
        sorted_roi = roi_by_channel.sort_values(by='ROI', ascending=False)
        first_highest = f"{sorted_roi['Channel'].iloc[0].title()}: {sorted_roi['ROI'].iloc[0]:.2f}X"
        second_highest = f"{sorted_roi['Channel'].iloc[1].title()}: {sorted_roi['ROI'].iloc[1]:.2f}X"
        third_highest = f"{sorted_roi['Channel'].iloc[2].title()}: {sorted_roi['ROI'].iloc[2]:.2f}X"
        fig = px.bar(roi_by_channel, x='Channel', y='ROI', color='Channel', title='ROI by Channel')
    elif type_of_analysis == "Customer Lifetime Value" :
        cltv_by_channel = data.groupby('Channel')['CLTV'].mean().reset_index()
        sorted_cltv = cltv_by_channel.sort_values(by='CLTV', ascending=False)
        first_highest = f"{sorted_cltv['Channel'].iloc[0].title()}: $ {sorted_cltv['CLTV'].iloc[0]:.2f}"
        second_highest = f"{sorted_cltv['Channel'].iloc[1].title()}: $ {sorted_cltv['CLTV'].iloc[1]:.2f}"
        third_highest = f"{sorted_cltv['Channel'].iloc[2].title()}: $ {sorted_cltv['CLTV'].iloc[2]:.2f}"
        fig = px.bar(cltv_by_channel, x='Channel', y='CLTV', color='Channel', title='CLTV by Channel')
        
    with placeholder.container():
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric(label="1st Channel 🥇", value= (first_highest))
        kpi2.metric(label="2nd Channel 🥈", value= (second_highest))
        kpi3.metric(label="3rd Channel 🥉", value= (third_highest))

        
        st.markdown("### :blue[Bar Chart] 📊")
      

        st.write(fig)
        
        st.markdown("### :blue[Detailed Data View] 🔍")
        st.dataframe(data, use_container_width=True, hide_index=True)
        time.sleep(0.5)