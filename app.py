import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Smart Bus System", layout="wide")

st.title("🚌 Smart Bus Crowd & Delay Prediction")

# Load data
df = pd.read_csv("bus_data.csv")

# Show data
st.subheader("📊 Bus Data")
st.dataframe(df)

# Chart
st.subheader("📈 Passenger Trend")
fig, ax = plt.subplots()
df.groupby("Time")["Passengers"].mean().plot(ax=ax)
st.pyplot(fig)

# Model
X = df[["Delay_Minutes"]]
y = df["Passengers"]

model = LinearRegression()
model.fit(X, y)

# Sidebar input
st.sidebar.header("Predict Crowd")

delay = st.sidebar.slider("Delay (minutes)", 0, 30)

pred = model.predict([[delay]])

# Output
st.subheader("🔮 Predicted Crowd Level")

if pred[0] < 40:
    st.success("🟢 Low Crowd")
elif pred[0] < 80:
    st.warning("🟡 Medium Crowd")
else:
    st.error("🔴 High Crowd")

st.write(f"Estimated Passengers: {int(pred[0])}")