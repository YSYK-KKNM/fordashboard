import streamlit as st
import pandas as pd
import openpyxl
st.title("Project Dashboard")
st.write("Main outputs of both individual project and group project are demonstrated as follows.")

st.latex(r"Y = \beta_0 + \beta_1X + \varepsilon")

col1, col2, col3 = st.columns(3)
col1.metric("Temperature","70 °F","1.2 °F")
col2.metric("Wind","9 mph","-8%")
col3.metric("Humidity","86%","4%")

st.markdown('<p style="font-size:20px; font-family:\"Times New Roman\", serif; color:#333333e;">1. Country $\mathrm{CO}_2$ Emissions per Year Over Time</p>', unsafe_allow_html=True)
st.markdown('<p style="font-size:16px; font-family:\"Times New Roman\", serif; color:#333333; line-height:1.6;">You may select a country to highlight by pressing the button</p>', unsafe_allow_html=True)
import numpy as np
@st.cache_data
def load_data():
    co2 = pd.read_excel('https://raw.githubusercontent.com/YSYK-KKNM/groupproject/main/yearly_co2_emissions_1000_tonnes1.xlsx')
    gdp = pd.read_excel('https://raw.githubusercontent.com/YSYK-KKNM/groupproject/main/GDP_growth.xlsx', skiprows=3)
    energy = pd.read_excel('https://raw.githubusercontent.com/YSYK-KKNM/groupproject/main/energy.xlsx', skiprows=3)
    disaster = pd.read_excel('https://raw.githubusercontent.com/YSYK-KKNM/groupproject/main/disaster.xlsx')
    temperature = pd.read_excel('https://raw.githubusercontent.com/YSYK-KKNM/groupproject/main/temperature.xlsx')
    
    return co2, gdp, energy, disaster, temperature

co2, gdp, energy, disaster, temperature = load_data()
co2= co2.melt(id_vars='country', var_name='Year', value_name='Emissions')
co2.rename(columns={'country': 'Country'}, inplace=True)
co2['Year']=pd.to_numeric(co2['Year'])
co2['Label']='CO2 Emissions (Metric Tons)'
co2=co2.rename(columns={'Emissions': 'Value'})
co2['Indicator']='Emissions'
co2['Value'] = co2['Value'].apply(lambda x:(float(str(x)[:-1])*1000 if str(x).endswith('k') else
                                            float(str(x)[:-1])*1000000 if str(x).endswith('M') else None))

gdp=gdp.melt(id_vars=['Country Name','Indicator Name'],var_name='Year',value_name='Value')
gdp['Year']=gdp['Year'].astype(int)
gdp=gdp.rename(columns={'Indicator Name':'Label', 'Country Name':'Country'})
gdp['Indicator']='GDP'

energy=energy.melt(id_vars=['Country Name','Indicator Name'],var_name='Year',value_name='Value')
energy=energy.rename(columns={'Country Name':'Country','Indicator Name':'Label'})
energy['Year']=energy['Year'].astype(int)
energy['Indicator']='Energy'

disaster['Date']=disaster['DisNo.'].astype(str)
disaster['Year']=disaster['Date'].str[:4].astype(int)
disaster=disaster.groupby(['Year','Disaster Type']).size().unstack(fill_value=0).reset_index()
disaster['Disaster']=disaster.drop(columns='Year').sum(axis=1)
disasters=disaster[['Year', 'Disaster']].copy()
disasters['Country']='Germany'
disasters= disasters.melt(id_vars=['Year', 'Country'],var_name='Indicator',value_name='Value')
disasters['Label']='Number of Disasters'

temperature['Date']=temperature['Date'].astype(str)
temperature['Year']=temperature['Date'].str[:4].astype(int)
temperature['Country']='Germany'
temperature['Indicator']='Temperature'
temperature['Label']='Temperature (Fahrenheit)'
temperature=temperature[['Year', 'Country', 'Indicator', 'Value', 'Label']]

combined=pd.concat([co2, gdp, energy, disasters, temperature], ignore_index=True)
combined['Region']=combined['Country'].apply(lambda x:'Germany' if x=='Germany' else 'Rest of the world')
combined=combined.dropna().sort_values(by='Country')

import matplotlib.pyplot as plt
us = st.button("USA")
ger = st.button("Germany")

if not us and not ger:
    fig, ax = plt.subplots(figsize=(12, 6))
    for country in co2['Country'].unique():
        xf = co2.loc[co2['Country'] == country]
        ax.plot(xf['Year'], xf['Value'], alpha=1,
                color='blue' if country == 'USA' else 'gray',
                linewidth=1.2 if country == 'USA' else 0.8,
                label='United States' if country == 'USA' else None)
    ax.set_title('Country $\mathrm{CO}_2$ Emissions per Year (1751–2019)', fontsize=16)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Emissions (Metric Tonnes)', fontsize=12)
    ax.legend(fontsize=12)
    ax.text(0.785, -0.114, 'Limited to reporting countries', transform=ax.transAxes, fontsize=12)
    ax.tick_params(labelsize=12)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
                
if us:
    fig, ax = plt.subplots(figsize=(12, 6))
    for country in co2['Country'].unique():
        xf = co2.loc[co2['Country'] == country]
        ax.plot(xf['Year'], xf['Value'], alpha=1,
                color='blue' if country == 'USA' else 'gray',
                linewidth=1.2 if country == 'USA' else 0.8,
                label='United States' if country == 'USA' else None)
    ax.set_title('Country $\mathrm{CO}_2$ Emissions per Year (1751–2019)', fontsize=16)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Emissions (Metric Tonnes)', fontsize=12)
    ax.legend(fontsize=12)
    ax.text(0.785, -0.114, 'Limited to reporting countries', transform=ax.transAxes, fontsize=12)
    ax.tick_params(labelsize=12)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
elif ger:
    fig, ax = plt.subplots(figsize=(12, 6))
    for country in co2['Country'].unique():
        xf = co2.loc[co2['Country'] == country]
        ax.plot(xf['Year'], xf['Value'], alpha=1,
                color='blue' if country == 'Germany' else 'gray',
                linewidth=1.2 if country == 'Germany' else 0.8,
                label='Germany' if country == 'Germany' else None)
    ax.set_title('Country $\mathrm{CO}_2$ Emissions per Year (1751–2019)', fontsize=16)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Emissions (Metric Tonnes)', fontsize=12)
    ax.legend(fontsize=12)
    ax.text(0.785, -0.114, 'Limited to reporting countries', transform=ax.transAxes, fontsize=12)
    ax.tick_params(labelsize=12)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
else:
    st.write("Please select a country to see the graph.")
