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
    utemperature=pd.read_csv('https://raw.githubusercontent.com/YSYK-KKNM/groupproject/main/temperature.csv', skiprows=4, na_values="-99")
    gtemperature = pd.read_excel('https://raw.githubusercontent.com/YSYK-KKNM/groupproject/main/temperature.xlsx')
    return co2, gdp, energy, disaster, utemperature, gtemperature
co2, gdp, energy, disaster, utemperature, gtemperature = load_data()

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

utemperature['Date'] = utemperature['Date'].astype(str)
utemperature['Year'] = utemperature['Date'].str[:4].astype(int)
utemperature['Country'] = 'USA'
utemperature['Indicator'] = 'Temperature'
utemperature['Label'] = 'Temperature (Fahrenheit)'
utemperature = utemperature[['Year', 'Country', 'Indicator', 'Value', 'Label']]

gtemperature['Date'] = gtemperature['Date'].astype(str)
gtemperature['Year'] = gtemperature['Date'].str[:4].astype(int)
gtemperature['Country'] = 'Germany'
gtemperature['Indicator'] = 'Temperature'
gtemperature['Label'] = 'Temperature (Fahrenheit)'
gtemperature = gtemperature[['Year', 'Country', 'Indicator', 'Value', 'Label']]

combined1 = pd.concat([co2, gdp, energy, disasters, utemperature], ignore_index=True)
combined1['Region'] = combined1['Country'].apply(lambda x: 'USA' if x == 'USA' else 'Rest of the world')
combined1 = combined1.dropna().sort_values(by='Country')
combined2 = pd.concat([co2, gdp, energy, disasters, gtemperature], ignore_index=True)
combined2['Region'] = combined2['Country'].apply(lambda x: 'Germany' if x == 'Germany' else 'Rest of the world')
combined2 = combined2.dropna().sort_values(by='Country')

d2019 = co2[co2['Year'] == 2019].copy()
d2019['rank'] = d2019['Value'].rank(ascending=False)
top = d2019[d2019['rank'] <= 10]
etop = co2[(co2['Country'].isin(top['Country'])) & (co2['Year'] >= 1900)].copy()

st.markdown("""
    <style>
        .css-1lcbk24 {
            background-color: red;
        }
    </style>
    """, unsafe_allow_html=True)

if "page" not in st.session_state:
    st.session_state.page = "b0"  
if "country_selected" not in st.session_state:
    st.session_state.country_selected = "USA"

def set_page(p):
    st.session_state.page = p

st.sidebar.title("Navigation")
st.sidebar.button("Introduction", on_click=set_page, args=("b0",))
st.sidebar.button("CO₂ Emissions per Year Over Time", on_click=set_page, args=("b1",))
st.sidebar.button("Top 10 Emissions-producing Countries", on_click=set_page, args=("b2",))
st.sidebar.button("Tile Plot of the Top 10 CO₂ Emission-producing Countries", on_click=set_page, args=("b3",))
st.sidebar.button("Distributions of Indicators by Year and Value", on_click=set_page, args=("b4",))
st.sidebar.button("Emissions&Temperature (USA)", on_click=set_page, args=("b5",))
st.sidebar.button("Emissions&Temperature/Natural Disasters (Germany)", on_click=set_page, args=("b6",))

page = st.session_state.page

if page=="b0":
    st.title("Project Dashboard")
    st.write("By Zebin You")
    st.markdown(
        '<p style="font-size:28px; font-family:\'Times New Roman\', serif; color:#333333;">'
        'Main outputs of both individual project and group project are demonstrated in this dashboard.'
        '</p>',
        unsafe_allow_html=True
    )
    st.write("Please select a module from the sidebar on the left.")

elif page == "b1":
    st.title("Country CO₂ Emissions per Year Over Time")
    cou = st.radio(
        "Select a country to view CO₂ Emissions per Year",
        ("USA", "Germany"),
        index=0 if st.session_state.country_selected == "USA" else 1,
        key="cou"
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    for country in co2["Country"].unique():
        xf = co2.loc[co2["Country"] == country]
        ax.plot(
            xf["Year"],
            xf["Value"],
            alpha=1,
            color="blue" if country == cou else "gray",
            linewidth=1.2 if country == cou else 0.8,
            label=cou if country == cou else None,
        )

    ax.set_title("Country $\\mathrm{CO}_2$ Emissions per Year (1751–2019)", fontsize=16)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Emissions (Metric Tonnes)", fontsize=12)
    ax.legend(fontsize=12)
    ax.text(0.785, -0.114, "Limited to reporting countries", transform=ax.transAxes, fontsize=12)
    ax.tick_params(labelsize=12)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

elif page=="b2":
    st.title('Top 10 Emissions-producing Countries (1900-2019)')
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

elif page=="b3":
    st.title('Tile Plot of the Top 10 CO₂ Emission-producing Countries')
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

elif page=="b4":
    st.title('Facet Figure: Distributions of Indicators by Year and Value')
    co = st.radio("Select a country", ("USA", "Germany"), index=0 if st.session_state.country_selected == 'USA' else 1)
    st.session_state.country_selected = co
    if st.session_state.country_selected == "USA":
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
    
    elif st.session_state.country_selected == "Germany":
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
        st.write("Please Select a Country First.")

elif page=="b5":
    st.title('Relationship Between Emissions and Temperature for USA')
    st.write("The Mean and Standard Deviation of CO₂ Emissions and Temperature")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Emissions Mean", "5142286")
    col2.metric("Emissions SD", "450549")
    col3.metric("Temperature Mean", "52.87°F")
    col4.metric("Temperature SD", "0.89°F")
    
    st.write("The Correlation Coefficient for CO₂ Emissions and Temperature")
    
    col5 = st.columns(1)
    col5[0].metric("Emissions & Temperature Correlation", "0.4712")
    
    us = combined1[(combined1['Country'] == 'USA') & (combined1['Year'].between(1900, 2024)) & (combined1['Indicator'].isin(['Emissions', 'Temperature']))]
    li_us = us.pivot(index='Year', columns='Indicator', values='Value').reset_index()
    df=li_us.copy()
    scaler=StandardScaler()
    df[['sc_emissions', 'sc_temperature']]=scaler.fit_transform(df[['Emissions', 'Temperature']])
    X=df['sc_emissions']
    y=df['sc_temperature']
    ##use model to conduct regression
    X=sm.add_constant(X)
    model=sm.OLS(y, X)
    results=model.fit()
    ## make the scatter plot and apply the regression line on it.
    y_sc=results.predict(X)
    plt.figure(figsize=(12, 6))
    plt.scatter(df['sc_emissions'], df['sc_temperature'], label='Standardized CO₂ Emissions', color='black', alpha=0.8)
    plt.plot(df['sc_emissions'], y_sc, color='blue', linewidth=2)
    plt.title('Us $\mathrm{CO}_2$ Emissions and Temperature (1980-2014)', fontsize=16)
    plt.xlabel('Scaled Emissions (Metric Tonnes)', fontsize=12)
    plt.ylabel('Scaled Temperature (Fahrenheit)', fontsize=12)
    plt.grid(alpha=0.3)    
    st.pyplot(plt)



elif page=="b6":
    st.title('Relationship Between Emissions and Temperature/Natural Disasters for Germany')
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
    st.write("The Correlation Coefficient for CO₂ Emissions and Each Type of Natural Disaster")
    colors = cm.viridis(np.linspace(0, 1, 7))
    corrs = {}
    gm=co2[(co2['Country']=='Germany')&(co2['Year'].between(1900,2024))]
    codi=pd.merge(disaster,gm,on='Year',how='inner')
    for col in disaster.columns[1:]: 
        corrs[col] = codi['Value'].corr(codi[col])
    cor=pd.DataFrame(list(corrs.items()), columns=['Disaster Type', 'Correlation with $\mathrm{CO}_2$ Emissions'])
    plt.figure(figsize=(12,6))
    plt.barh(cor['Disaster Type'],cor['Correlation with $\mathrm{CO}_2$ Emissions'], color=colors)
    plt.title('Correlation between $\mathrm{CO}_2$ Emissions and Disaster Types', fontsize=14)
    plt.xlabel('Correlation with $\mathrm{CO}_2$', fontsize=12)
    plt.ylabel('Disaster Type', fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(plt)
else: None
    
