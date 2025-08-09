import streamlit as st

st.title("My first dashboard")
st.write("Hello World!")

st.latex(r"Y = \beta_0 + \beta_1X + \varepsilon")

col1, col2, col3 = st.columns(3)
col1.metric("Temperature","70 °F","1.2 °F")
col2.metric("Wind","9 mph","-8%")
col3.metric("Humidity","86%","4%")

import numpy as np
seed_for_prng = 78557
prng = np.random.default_rng(seed_for_prng)  # prng=probabilistic random number generator
x = np.linspace(0, 8, 16)
y1 = 3 + 4*x/8 + prng.uniform(0.0, 0.5, len(x))
y2 = 1 + 2*x/8 + prng.uniform(0.0, 0.5, len(x))

import matplotlib.pyplot as plt
# usual code for plot
fig, ax = plt.subplots()
ax.fill_between(x, y1, y2, alpha=.5, linewidth=0)
ax.plot(x, (y1 + y2)/2, linewidth=2)
ax.set_title("Chart with forecast bands", loc="left")
ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
       ylim=(0, 8), yticks=np.arange(1, 8))
ax.set_xlabel("Time")
ax.set_ylabel("Value")
# Show the chart on the streamlit dashboard
st.pyplot(fig)

import pandas as pd
import altair as alt
df = pd.DataFrame(
     prng.standard_normal(size=(200, 3)),
     columns=['a', 'b', 'c'])
alt_chart = alt.Chart(df).mark_circle().encode(
     x='a', y='b', size='c', color='c', tooltip=['a', 'b', 'c']
     )
st.altair_chart(alt_chart, use_container_width=True)

import plotly.express as px
df = px.data.iris()
pxfig = px.scatter(df, x="sepal_width", y="sepal_length", color="species",
                 size='petal_length', hover_data=['petal_width'])
st.plotly_chart(pxfig, use_container_width=True)

if st.button('Say Hello'):
    st.write('Hello!')

genre = st.radio(
     "What's your favourite movie genre",
     ('Comedy', 'Drama', 'Documentary'))
if genre == (answer := 'Comedy'):
    st.write(f'You selected {answer}.')
elif genre == (answer := 'Drama'):
    st.write(f'You selected {answer}.')
else:
    st.write("You didn't select comedy or drama.")
