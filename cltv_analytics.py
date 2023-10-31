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

if type_of_analysis == "Conversion Rate" :
        data = data.rename(columns={data.columns[0]: 'Campaign_ID'})
channel_groups = data.groupby('Channel')

color_map = {
    'email marketing': 'rgb(125, 33, 191)',
    'paid advertising': 'rgb(224, 11, 54)',
    'referral': 'rgb(1, 207, 81)',
    'social media':'rgb(246, 41, 109)'}


for seconds in range(800):
    data['Cost'] = np.random.uniform(10, 20, len(data))
    data['Conversion_rate'] = np.random.uniform(0.12, 0.16, len(data))
    data['Revenue'] = np.random.uniform(36, 46, len(data))

    random_cost_changes = np.random.normal(0, 4, len(data))
    random_conversion_rate_changes = np.random.normal(0, 0.045, len(data))
    random_revenue_changes = np.random.normal(0, 4, len(data))

    # Update the data using vectorized operations
    data['Cost'] += random_cost_changes
    data['Conversion_rate'] += random_conversion_rate_changes
    data['Revenue'] += random_revenue_changes
    # Recalculate ROI and CLTV for each row after the changes
    data['ROI'] = (data['Revenue'] - data['Cost']) / data['Cost']
    data['CLTV'] = (data['Revenue'] - data['Cost']) 
    width = 700

    if type_of_analysis == "Customer Acquisition Cost" :
        cost_by_channel = channel_groups['Cost'].mean().reset_index()
        sorted_costs = cost_by_channel.sort_values(by='Cost', ascending=True)
        first_highest = f"{sorted_costs['Channel'].iloc[0].title()}: $ {sorted_costs['Cost'].iloc[0]:.2f}"
        second_highest = f"{sorted_costs['Channel'].iloc[1].title()}: $ {sorted_costs['Cost'].iloc[1]:.2f}"
        third_highest = f"{sorted_costs['Channel'].iloc[2].title()}: $ {sorted_costs['Cost'].iloc[2]:.2f}"
        latest_customers = data.tail(20)
        fig1 = px.bar(cost_by_channel, x='Channel', y='Cost', color='Channel', color_discrete_map=color_map, labels={'Cost': 'Customer Acquisition Cost'}, title='Customer Acquisition Cost by Channel', width=width).update_yaxes(tickprefix="$")
        fig2  = px.line(latest_customers, x=latest_customers.index, y='Cost', title='Cost to Acquire - Last 20 Customers (All Channels)', width=width).update_xaxes(showticklabels=False).update_xaxes(title_text='').update_yaxes(tickprefix="$")       
    elif type_of_analysis == "Conversion Rate" :
        conversion_rate_by_channel = channel_groups['Conversion_rate'].mean().reset_index()
        sorted_conversation_rates = conversion_rate_by_channel.sort_values(by='Conversion_rate', ascending=False)
        first_highest = f"{sorted_conversation_rates['Channel'].iloc[0].title()}: {sorted_conversation_rates['Conversion_rate'].iloc[0]*100:.2f}%"
        second_highest = f"{sorted_conversation_rates['Channel'].iloc[1].title()}: {sorted_conversation_rates['Conversion_rate'].iloc[1]*100:.2f}%"
        third_highest = f"{sorted_conversation_rates['Channel'].iloc[2].title()}: {sorted_conversation_rates['Conversion_rate'].iloc[2]*100:.2f}%"
        latest_customers = data.tail(20)
        fig1 = px.bar(conversion_rate_by_channel, x='Channel', y='Conversion_rate', color='Channel',color_discrete_map=color_map, title='Average Conversion Rate by Channel').update_yaxes(tickformat=".0%")
        fig2  = px.pie(conversion_rate_by_channel, values='Conversion_rate', names='Channel', color='Channel', color_discrete_map=color_map,title='Percentage of Customers Converted by Channel', width=width)
    elif type_of_analysis == "ROI" :
        roi_by_channel = data.groupby('Channel')['ROI'].mean().reset_index()
        sorted_roi = roi_by_channel.sort_values(by='ROI', ascending=False)
        first_highest = f"{sorted_roi['Channel'].iloc[0].title()}: {sorted_roi['ROI'].iloc[0]:.2f}X"
        second_highest = f"{sorted_roi['Channel'].iloc[1].title()}: {sorted_roi['ROI'].iloc[1]:.2f}X"
        third_highest = f"{sorted_roi['Channel'].iloc[2].title()}: {sorted_roi['ROI'].iloc[2]:.2f}X"
        latest_customers = data.tail(20)
        fig1 = px.bar(roi_by_channel, x='Channel', y='ROI', color='Channel', title='Average ROI by Channel', color_discrete_map=color_map, width=width).update_yaxes(ticksuffix="X")
        fig2  = px.line(latest_customers, x=latest_customers.index, y='ROI', title='ROI - Last 20 Customers (All Channels)', width=width).update_xaxes(showticklabels=False).update_xaxes(title_text='').update_yaxes(ticksuffix="X")
    elif type_of_analysis == "Customer Lifetime Value" :
        cltv_by_channel = data.groupby('Channel')['CLTV'].mean().reset_index()
        sorted_cltv = cltv_by_channel.sort_values(by='CLTV', ascending=False)
        first_highest = f"{sorted_cltv['Channel'].iloc[0].title()}: $ {sorted_cltv['CLTV'].iloc[0]:.2f}"
        second_highest = f"{sorted_cltv['Channel'].iloc[1].title()}: $ {sorted_cltv['CLTV'].iloc[1]:.2f}"
        third_highest = f"{sorted_cltv['Channel'].iloc[2].title()}: $ {sorted_cltv['CLTV'].iloc[2]:.2f}"
        latest_customers = data.tail(20)
        fig1 = px.bar(cltv_by_channel, x='Channel', y='CLTV', color='Channel', title='Average CLTV by Channel',color_discrete_map=color_map).update_yaxes(tickprefix="$")
        fig2  = px.line(latest_customers, x=latest_customers.index, y='CLTV', title='CLTV - Last 20 Customers (All Channels)', width=width).update_xaxes(showticklabels=False).update_xaxes(title_text='').update_yaxes(tickprefix="$")
        
    with placeholder.container():
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric(label="1st Channel 🥇", value= (first_highest))
        kpi2.metric(label="2nd Channel 🥈", value= (second_highest))
        kpi3.metric(label="3rd Channel 🥉", value= (third_highest))

        fig_col1, fig_col2= st.columns(2)
        
        with fig_col1:
            st.markdown("### :blue[Bar Chart] 📊")
            st.write(fig1)
    
        with fig_col2:
            st.markdown("### :blue[Line Chart] 📈")
            st.write(fig2)
        

        
 
        
        st.markdown("### :blue[Detailed Data View] 🔍")
        st.dataframe(data, use_container_width=True, hide_index=True)
        time.sleep(0.5)
        