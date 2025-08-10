import streamlit as st
import pandas as pd
import openpyxl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import seaborn as sns

@st.cache_data
def load_data():
    co2 = pd.read_excel('https://raw.githubusercontent.com/YSYK-KKNM/groupproject/main/yearly_co2_emissions_1000_tonnes1.xlsx')
    gdp = pd.read_excel('https://raw.githubusercontent.com/YSYK-KKNM/groupproject/main/GDP_growth.xlsx', skiprows=3)
    energy = pd.read_excel('https://raw.githubusercontent.com/YSYK-KKNM/groupproject/main/energy.xlsx', skiprows=3)
    disaster = pd.read_excel('https://raw.githubusercontent.com/YSYK-KKNM/groupproject/main/disaster.xlsx')
    temperature = pd.read_excel('https://raw.githubusercontent.com/YSYK-KKNM/groupproject/main/temperature.xlsx')
    return co2, gdp, energy, disaster, temperature
co2, gdp, energy, disaster, temperature = load_data()

co2 = co2.melt(id_vars='country', var_name='Year', value_name='Emissions')
co2.rename(columns={'country': 'Country'}, inplace=True)
co2['Year'] = pd.to_numeric(co2['Year'])
co2['Label'] = 'CO2 Emissions (Metric Tons)'
co2 = co2.rename(columns={'Emissions': 'Value'})
co2['Indicator'] = 'Emissions'
co2['Value'] = co2['Value'].apply(lambda x: (float(str(x)[:-1]) * 1000 if str(x).endswith('k') else
                                              float(str(x)[:-1]) * 1000000 if str(x).endswith('M') else None))

gdp = gdp.melt(id_vars=['Country Name', 'Indicator Name'], var_name='Year', value_name='Value')
gdp['Year'] = gdp['Year'].astype(int)
gdp = gdp.rename(columns={'Indicator Name': 'Label', 'Country Name': 'Country'})
gdp['Indicator'] = 'GDP'

energy = energy.melt(id_vars=['Country Name', 'Indicator Name'], var_name='Year', value_name='Value')
energy = energy.rename(columns={'Country Name': 'Country', 'Indicator Name': 'Label'})
energy['Year'] = energy['Year'].astype(int)
energy['Indicator'] = 'Energy'

disaster['Date'] = disaster['DisNo.'].astype(str)
disaster['Year'] = disaster['Date'].str[:4].astype(int)
disaster = disaster.groupby(['Year', 'Disaster Type']).size().unstack(fill_value=0).reset_index()
disaster['Disaster'] = disaster.drop(columns='Year').sum(axis=1)
disasters = disaster[['Year', 'Disaster']].copy()
disasters['Country'] = 'Germany'
disasters = disasters.melt(id_vars=['Year', 'Country'], var_name='Indicator', value_name='Value')
disasters['Label'] = 'Number of Disasters'

temperature['Date'] = temperature['Date'].astype(str)
temperature['Year'] = temperature['Date'].str[:4].astype(int)
temperature['Country'] = 'Germany'
temperature['Indicator'] = 'Temperature'
temperature['Label'] = 'Temperature (Fahrenheit)'
temperature = temperature[['Year', 'Country', 'Indicator', 'Value', 'Label']]

combined1 = pd.concat([co2, gdp, energy, disasters, temperature], ignore_index=True)
combined1['Region'] = combined1['Country'].apply(lambda x: 'USA' if x == 'USA' else 'Rest of the world')
combined1 = combined1.dropna().sort_values(by='Country')
combined2 = pd.concat([co2, gdp, energy, disasters, temperature], ignore_index=True)
combined2['Region'] = combined2['Country'].apply(lambda x: 'Germany' if x == 'Germany' else 'Rest of the world')
combined2 = combined2.dropna().sort_values(by='Country')

d2019 = co2[co2['Year'] == 2019].copy()
d2019['rank'] = d2019['Value'].rank(ascending=False)
top = d2019[d2019['rank'] <= 10]
etop = co2[(co2['Country'].isin(top['Country'])) & (co2['Year'] >= 1900)].copy()

st.title("Project Dashboard")
st.write("Main outputs of both individual project and group project are demonstrated as follows.")
st.sidebar.title("Navigation")
b1 = st.sidebar.button("CO₂ Emissions per Year Over Time")
b2 = st.sidebar.button("Top 10 Emissions-producing Countries")
b3 = st.sidebar.button("Tile Plot of the Top 10 CO₂ Emission-producing Countries")
b4 = st.sidebar.button("Distributions of Indicators by Year and Value")
b5 = st.sidebar.button("Emissions&Temperature (USA)")
b6 = st.sidebar.button("Emissions&Temperature/Natural Disasters (Germany)")

st.markdown("""
    <style>
        .css-1lcbk24 {
            background-color: red;
        }
    </style>
    """, unsafe_allow_html=True)

if b1:
    st.markdown('<p style="font-size:20px; font-family:\"Times New Roman\", serif; color:#333333e;">1. Country CO₂ Emissions per Year Over Time</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:16px; font-family:\"Times New Roman\", serif; color:#333333; line-height:1.6;">You may select a country to highlight by pressing the button</p>', unsafe_allow_html=True)

    co= st.radio(
        "Select a country to view CO₂ Emissions per Year",
        ('USA', 'Germany')
    )
    
    if co== 'USA':
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
    
    elif co == 'Germany':
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
    else: None


elif b2:
    st.markdown('<p style="font-size:20px; font-family:\"Times New Roman\", serif; color:#333333e;">2. Top 10 Emissions-producing Countries (1900-2019)</p>', unsafe_allow_html=True)
    cns = etop['Country'].unique()
    colors = cm.viridis(np.linspace(0, 1, len(cns)))
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, country in enumerate(cns):
        yf = etop.loc[etop['Country'] == country]
        ax.plot(yf['Year'], yf['Value'], color=colors[i], linewidth=1, alpha=1, label=country)
        ax.text(yf['Year'].iloc[-1] - 3.5, yf['Value'].iloc[-1], country, fontsize=12, color=colors[i])
    ax.set_title('Top 10 Emissions-producing Countries (1900-2019)', fontsize=16)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Emissions (Metric Tonnes)', fontsize=12)
    ax.text(0.001, 0.96, 'Ordered by Emissions Produced in 2019', transform=ax.transAxes, fontsize=12)
    ax.tick_params(labelsize=12)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

elif b3:
    st.markdown('<p style="font-size:20px; font-family:\"Times New Roman\", serif; color:#333333e;">3. Tile Plot of the Top 10 CO₂ Emission-producing Countries</p>', unsafe_allow_html=True)
    etop['loge'] = np.log(etop['Value'])
    tp = top[['Country', 'rank']]
    etop = etop.merge(tp, on='Country', how='left')
    etop.sort_values(['rank', 'Year'], inplace=True)
    mapdata = etop.pivot(index='Country', columns='Year', values='loge')
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(mapdata, cmap='viridis', cbar_kws={'label': 'Ln($\mathrm{CO}_2$ Emissions)'}, xticklabels=5)
    ax.text(0.24, 1.08, "Top 10 $\mathrm{CO}_2$ Emission-producing Countries", fontsize=16, transform=ax.transAxes)
    ax.text(0.001, 1.03, "Ordered by Emissions Produced in 2019", fontsize=12, transform=ax.transAxes)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Country', fontsize=12)
    plt.xticks()
    plt.tight_layout()
    st.pyplot(fig)

elif b4:
    st.markdown('<p style="font-size:20px; font-family:\"Times New Roman\", serif; color:#333333e;">4. Facet Figure: Distributions of Indicators by Year and Value</p>', unsafe_allow_html=True)
    co=st.radio("Select a country", ("USA", "Germany"))
    if co=="USA":
        fig, axes = plt.subplots(3, 2, figsize=(14, 9), sharex='col', sharey='row')
        indicators = ['Emissions', 'Energy', 'GDP']
        regions = ['Rest of the world', 'USA']
        xl = (1750, 2025)
        yl = []
        for indicator in indicators:
            lm = combined1[combined1['Indicator'] == indicator]
            yl.append((lm['Value'].min(), lm['Value'].max()))
        for i, indicator in enumerate(indicators):
            for j, region in enumerate(regions):
                ax = axes[i, j]
                zf = combined1[(combined1['Indicator'] == indicator) & (combined1['Region'] == region)]
                for country in zf['Country'].unique():
                    wf = zf[zf['Country'] == country]
                    sns.lineplot(data=wf, x='Year', y='Value', ax=ax, color='black', linewidth=0.8, alpha=0.7)
                ax.set_xlim(xl)
                ax.set_ylim(yl[i])
                if i == 0:
                    ax.set_title(region, fontsize=14)
                if j == 0:
                    ax.set_ylabel(indicator, fontsize=12)
                if i == 2:
                    ax.set_xlabel('Year', fontsize=11)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.suptitle('Distribution of Indicators by Year and Value (USA)', fontsize=16)
        st.pyplot(fig)
    
    elif co=="Germany":
        fig, axes = plt.subplots(3, 2, figsize=(14, 9), sharex='col', sharey='row')
        indicators = ['Emissions', 'Energy', 'GDP']
        regions = ['Rest of the world', 'Germany']
        xl = (1750, 2025)
        yl = []
        for indicator in indicators:
            lm = combined2[combined2['Indicator'] == indicator]
            yl.append((lm['Value'].min(), lm['Value'].max()))
        for i, indicator in enumerate(indicators):
            for j, region in enumerate(regions):
                ax = axes[i, j]
                zf = combined2[(combined2['Indicator'] == indicator) & (combined2['Region'] == region)]
                for country in zf['Country'].unique():
                    wf = zf[zf['Country'] == country]
                    sns.lineplot(data=wf, x='Year', y='Value', ax=ax, color='black', linewidth=0.8, alpha=0.7)
                ax.set_xlim(xl)
                ax.set_ylim(yl[i])
                if i == 0:
                    ax.set_title(region, fontsize=14)
                if j == 0:
                    ax.set_ylabel(indicator, fontsize=12)
                if i == 2:
                    ax.set_xlabel("Year", fontsize=11)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.suptitle("Distribution of Indicators by Year and Value (Germany)", fontsize=16)
        st.pyplot(fig)
    else:
        None


elif b5:
    st.markdown('<p style="font-size:20px; font-family:\"Times New Roman\", serif; color:#333333e;">Relationship Between Emissions and Temperature for USA</p>', unsafe_allow_html=True)
    st.write("The Mean and Standard Deviation of CO₂ Emissions and Temperature")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Emissions Mean", "5142286")
    col2.metric("Emissions SD", "450549")
    col3.metric("Temperature Mean", "52.87°F")
    col4.metric("Temperature SD", "0.89°F")
    st.write("The Correlation Coefficient for CO₂ Emissions and Temperature")
    col5 = st.columns(1)
    col5[0].metric("Emissions&Temperature", "0.4712")
    us = combined1[(combined1['Country'] == 'USA') & (combined1['Year'].between(1900, 2024)) & (combined1['Indicator'].isin(['Emissions', 'Temperature']))]
    li_us = us.pivot(index='Year', columns='Indicator', values='Value').reset_index()
    df = li_us.copy()
    scaler = StandardScaler()
    df[['sc_emissions', 'sc_temperature']] = scaler.fit_transform(df[['Emissions', 'Temperature']])
    x = df['sc_emissions']
    y = df['sc_temperature']
    tocl = pd.concat([x, y], axis=1)
    clean = tocl.dropna()
    x = clean['sc_emissions']
    y = clean['sc_temperature']
    x = sm.add_constant(x)
    model = sm.OLS(y, x)
    results = model.fit()
    y_sc = results.predict(x)
    plt.figure(figsize=(12, 6))
    plt.scatter(x['sc_emissions'], y, label='Standardized CO₂ Emissions', color='black', alpha=0.8)
    plt.plot(x['sc_emissions'], y_sc, color='blue', linewidth=2)
    plt.title('Germany $\mathrm{CO}_2$ Emissions and Temperature (1980-2024)', fontsize=16)
    plt.xlabel('Scaled Emissions (Metric Tonnes)', fontsize=12)
    plt.ylabel('Scaled Temperature (Fahrenheit)', fontsize=12)
    plt.grid(alpha=0.3)
    st.pyplot(plt)

elif b6:
    st.markdown('<p style="font-size:20px; font-family:\"Times New Roman\", serif; color:#333333e;">Relationship Between Emissions and Temperature/Natural Disasters for Germany</p>', unsafe_allow_html=True)
    st.write("The Mean and Standard Deviation of CO₂ Emissions and Temperature")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Emissions Mean", "716447")
    col2.metric("Emissions SD", "257147")
    col3.metric("Temperature Mean", "47.75°F")
    col4.metric("Temperature SD", "1.6°F")
    st.write("The Correlation Coefficient for CO₂ Emissions and Temperature/Natural Disasters")
    col5, col6 = st.columns(2)
    col5.metric("Emissions&Temperature", "0.2013")
    col6.metric("Emissions&Natural Disasters", "0.0438")
    ger = combined2[(combined2['Country'] == 'Germany') & (combined2['Year'].between(1900, 2024)) & (combined2['Indicator'].isin(['Emissions', 'Temperature']))]
    li_ger = ger.pivot(index='Year', columns='Indicator', values='Value').reset_index()
    df = li_ger.copy()
    scaler = StandardScaler()
    df[['sc_emissions', 'sc_temperature']] = scaler.fit_transform(df[['Emissions', 'Temperature']])
    x = df['sc_emissions']
    y = df['sc_temperature']
    tocl = pd.concat([x, y], axis=1)
    clean = tocl.dropna()
    x = clean['sc_emissions']
    y = clean['sc_temperature']
    x = sm.add_constant(x)
    model = sm.OLS(y, x)
    results = model.fit()
    y_sc = results.predict(x)
    plt.figure(figsize=(12, 6))
    plt.scatter(x['sc_emissions'], y, label='Standardized CO₂ Emissions', color='black', alpha=0.8)
    plt.plot(x['sc_emissions'], y_sc, color='blue', linewidth=2)
    plt.title('Germany $\mathrm{CO}_2$ Emissions and Temperature (1980-2024)', fontsize=16)
    plt.xlabel('Scaled Emissions (Metric Tonnes)', fontsize=12)
    plt.ylabel('Scaled Temperature (Fahrenheit)', fontsize=12)
    plt.grid(alpha=0.3)
    st.pyplot(plt)
else: 
    st.write("Select a module on the left first.")
    
