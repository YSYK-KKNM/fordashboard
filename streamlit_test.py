import streamlit as st
import pandas as pd
import openpyxl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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

combined1=pd.concat([co2, gdp, energy, disasters, temperature], ignore_index=True)
combined1['Region']=combined1['Country'].apply(lambda x:'USA' if x=='USA' else 'Rest of the world')
combined1=combined1.dropna().sort_values(by='Country')
combined2=pd.concat([co2, gdp, energy, disasters, temperature], ignore_index=True)
combined2['Region']=combined2['Country'].apply(lambda x:'Germany' if x=='Germany' else 'Rest of the world')
combined2=combined2.dropna().sort_values(by='Country')

st.title("Project Dashboard")
st.write("Main outputs of both individual project and group project are demonstrated as follows.")
st.markdown("""
    <style>
        .css-1d391kg {
            background-color: red;
        }
    </style>
    """, unsafe_allow_html=True)
st.sidebar.title("Navigation")
b1 = st.sidebar.button("Variations of CO₂ Emissions Over Time")
b2 = st.sidebar.button("Relationship Between CO₂ Emissions and Temperature or Natural Disasters")
if b1:
    ##1
    st.markdown('<p style="font-size:20px; font-family:\"Times New Roman\", serif; color:#333333e;">1. Country CO₂ Emissions per Year Over Time</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:16px; font-family:\"Times New Roman\", serif; color:#333333; line-height:1.6;">You may select a country to highlight by pressing the button</p>', unsafe_allow_html=True)
    
    us1 = st.button("USA", key='us1')
    ger1 = st.button("Germany", key='ger1')
    
    if not us1 and not ger1:
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
                    
    elif us1:
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
    elif ger1:
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
    ##2
    st.markdown('<p style="font-size:20px; font-family:\"Times New Roman\", serif; color:#333333e;">2. Top 10 Emissions-producing Countries (1900-2019)</p>', unsafe_allow_html=True)
    d2019=co2[co2['Year']==2019].copy()
    d2019['rank']=d2019['Value'].rank(ascending=False)
    top=d2019[d2019['rank']<=10]
    etop=co2[(co2['Country'].isin(top['Country']))&(co2['Year']>=1900)].copy()
    
    cns=etop['Country'].unique()
    colors=cm.viridis(np.linspace(0,1,len(cns)))             
    fig,ax=plt.subplots(figsize=(12,6))
    for i,country in enumerate(cns):
        yf=etop.loc[etop['Country']==country]
        ax.plot(yf['Year'],yf['Value'],color=colors[i],linewidth=1,alpha=1,label=country)
        ax.text(yf['Year'].iloc[-1]-3.5, yf['Value'].iloc[-1],country, fontsize=12,color=colors[i])
    ##add title,labels,legends for the plot    
    ax.set_title('Top 10 Emissions-producing Countries (1900-2019)',fontsize=16)
    ax.set_xlabel('Year',fontsize=12)
    ax.set_ylabel('Emissions (Metric Tonnes)',fontsize=12)
    ax.text(0.001,0.96, 'Ordered by Emissions Produced in 2019',transform=ax.transAxes,fontsize=12)
    ax.tick_params(labelsize=12)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    
    ##3
    st.markdown('<p style="font-size:20px; font-family:\"Times New Roman\", serif; color:#333333e;">3. Tile Plot of the Top 10 CO₂ Emission-producing Countries</p>', unsafe_allow_html=True)
    etop['loge']=np.log(etop['Value'])
    tp=top[['Country','rank']]
    etop=etop.merge(tp,on='Country',how='left')
    etop.sort_values(['rank', 'Year'],inplace=True)
    mapdata=etop.pivot(index='Country',columns='Year',values='loge')
    import seaborn as sns
    fig,ax=plt.subplots(figsize=(12,6))
    sns.heatmap(mapdata,cmap='viridis',cbar_kws={'label':'Ln($\mathrm{CO}_2$ Emissions)'},xticklabels=5)
    ##In this part of code, we can use ax.text to insert title in case it overlaps the sentence under it.
    ax.text(0.24, 1.08, "Top 10 $\mathrm{CO}_2$ Emission-producing Countries", fontsize=16, transform=ax.transAxes)
    ax.text(0.001, 1.03, "Ordered by Emissions Produced in 2019", fontsize=12, transform=ax.transAxes)
    ax.set_xlabel('Year',fontsize=12)
    ax.set_ylabel('Country',fontsize=12)
    plt.xticks()
    plt.tight_layout()
    st.pyplot(fig)
    
    ##4
    st.markdown('<p style="font-size:20px; font-family:\"Times New Roman\", serif; color:#333333e;">4. Facet Figure: Distributions of Indicators by Year and Value</p>', unsafe_allow_html=True)
    us2 = st.button("USA",key='us2')
    ger2 = st.button("Germany",key='ger2')
    
    if not us2 and not ger2:
        st.write('Please Select a Country First.')
    elif us2:
        fig,axes=plt.subplots(3,2, figsize=(14, 9), sharex='col',sharey='row')
        indicators=['Emissions', 'Energy', 'GDP']
        regions=['Rest of the world', 'USA']
        xl=(1750, 2025)
        yl=[]
        for indicator in indicators:
            lm=combined1[combined1['Indicator']==indicator]
            yl.append((lm['Value'].min(), lm['Value'].max()))
        for i, indicator in enumerate(indicators):
            for j, region in enumerate(regions):
                ax=axes[i, j]
                zf=combined1[(combined1['Indicator']==indicator)&(combined1['Region']==region)]
                for country in zf['Country'].unique():
                    wf=zf[zf['Country']==country]
                    sns.lineplot(data=wf, x='Year', y='Value', ax=ax,color='black', linewidth=0.8, alpha=0.7)
                ax.set_xlim(xl)
                ax.set_ylim(yl[i]) 
                if i==0:
                    ax.set_title(region, fontsize=14,)
                if j==0:
                    ax.set_ylabel(indicator, fontsize=12)
                if i==2:
                    ax.set_xlabel('Year', fontsize=11)        
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.suptitle('Distribution of Indicators by Year and Value', fontsize=16)
        st.pyplot(fig)
    elif ger2:
        fig,axes=plt.subplots(3,2, figsize=(14, 9), sharex='col',sharey='row')
        indicators=['Emissions', 'Energy', 'GDP']
        regions=['Rest of the world', 'Germany']
        xl=(1750, 2025)
        yl=[]
        for indicator in indicators:
            lm=combined2[combined2['Indicator']==indicator]
            yl.append((lm['Value'].min(), lm['Value'].max()))
        for i, indicator in enumerate(indicators):
            for j, region in enumerate(regions):
                ax=axes[i, j]
                zf=combined2[(combined2['Indicator']==indicator)&(combined2['Region']==region)]
                for country in zf['Country'].unique():
                    wf=zf[zf['Country']==country]
                    sns.lineplot(data=wf, x='Year', y='Value', ax=ax,color='black', linewidth=0.8, alpha=0.7)
                ax.set_xlim(xl)
                ax.set_ylim(yl[i]) 
                if i==0:
                    ax.set_title(region, fontsize=14,)
                if j==0:
                    ax.set_ylabel(indicator, fontsize=12)
                if i==2:
                    ax.set_xlabel("Year", fontsize=11)        
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.suptitle("Distribution of Indicators by Year and Value", fontsize=16)
        st.pyplot(fig)
