import pandas as pd
import numpy as np
import os

import streamlit as st
import joblib

st.set_page_config(page_title="Mod Predictions", layout="wide")


st.title(" Flight Price Prediction")
st.sidebar.markdown("""
<style>
    .big-link {
        font-size: 24px;
        font-weight: bold;
        color: #1f77b4;
        text-decoration: underline;
        margin-bottom: 15px;
        display: block;
        cursor: pointer;
    }
    .big-link:hover {
        color: #d62728;
    }
</style>

<a href="#introduction" class="big-link">Introduction</a>
<a href="#project-overview" class="big-link">Project Overview</a>
<a href="#prediction" class="big-link">Prediction</a>
""", unsafe_allow_html=True)




st.markdown('<h2 id="introduction"> Introduction</h2>', unsafe_allow_html=True)
st.write("""Air travel pricing is dynamic and influenced by multiple factors such as the airline, number of stops, travel duration, and seasonal demand. 
         In this project, we explore and analyze flight fare data to uncover trends, relationships, and patterns that impact ticket costs. 
         By combining data exploration with predictive modeling""")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("https://www.istockphoto.com/photo/airplane-flying-over-tropical-sea-at-sunset-gm1526986072-524905290", width=400)

st.markdown("Model trained on below input data to perform  predictions.")



st.markdown('<h2 id="project-overview"> Project Overview</h2>', unsafe_allow_html=True)

# Objective
st.header(" Objective")
st.write("""The main objective of this project is to analyze and model flight fare data to identify the factors that significantly influence ticket prices.Exploring the relationship between features such as airline, source, destination, number of stops, and journey duration.
         Identifying patterns and trends that can help in predicting flight fares.
         Building a predictive model that can estimate ticket prices with reasonable accuracy.
         Providing actionable insights for travelers, airline operators, and travel agencies to make data-driven decisions.""")

column_data = {"Airline": "Name of the airline operating the flight",
    "Date_of_Journey": "Scheduled departure date of the journey",
    "Source": "City of departure",
    "Destination": "City of arrival",
    "Route": "Path taken by the flight, including stopovers",
    "Dep_Time": "Scheduled departure time",
    "Arrival_Time": "Scheduled arrival time",
    "Duration": "Total travel time from departure to arrival",
    "Total_Stops": "Number of stopovers between source and destination",
    "Additional_Info": "Miscellaneous flight information",
    "Price": "Ticket price (target variable)"}

column_data = pd.DataFrame(list(column_data.items()), columns=["Column Name", "Description"])
st.dataframe(column_data)


st.markdown('<h2 id="prediction">🐾 Prediction</h2>', unsafe_allow_html=True)
st.write("---")


base_dir = os.path.dirname(os.path.abspath(__file__))

csv_path = os.path.join(base_dir, "input.csv")
model_path = os.path.join(base_dir, "ridge_model.pkl")

Data = pd.read_csv(csv_path)
model = joblib.load(model_path)


st.write("Ridge algorithm applied to the below Dataset:")
st.dataframe(Data.head())
#st.write("---")

st.subheader(":blue[ Provide details to Discover Flight prices:]")



col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    airlines = st.selectbox("Select airlines:", Data.Airline.unique()[:5])
with col2:
    Source = st.selectbox("Select source:", Data.Source.unique())
with col3:
    Destination = st.selectbox("Select destination:", Data.Destination.unique())

col4, col5, col6 = st.columns([1, 1, 1])
with col4:
    flight_duration = st.number_input("Enter flight duration in minutes:")
with col5:
    stops = st.number_input("Enter number of stops:")
with col6:
    month_number = st.number_input("Enter month:")


col7, col8, col9 = st.columns([1, 1, 1])
with col7:
    day = st.number_input("Select date:")
with col8:
    deperture_hour = st.number_input("Select departure hour:")
with col9:
    deperture_minute = st.number_input("Select deperture minute:")

col10, col11 = st.columns([1, 1])
with col10:
    Arrival_hour = st.number_input("enter arrival hour :")
with col11:
    Arrival_minute = st.number_input("enter arrival minute:")

if st.button("Predict"):
    user_data = pd.DataFrame([[airlines,Source,Destination,flight_duration,stops,month_number,
                               day,deperture_hour,deperture_minute,Arrival_hour,Arrival_minute]],
                             columns=Data.columns)

    st.write("Given Data:")
    st.dataframe(user_data)

    user_data.replace({
            'Trujet': 1, 'SpiceJet': 2, 'Air Asia': 3, 'IndiGo': 4, 'GoAir': 5, 
            'Vistara': 6, 'Vistara Premium economy': 7, 'Air India': 8, 
            'Multiple carriers': 9, 'Multiple carriers Premium economy': 10, 
            'Jet Airways': 11, 'Jet Airways Business': 12,
            'Chennai': 1, 'Mumbai': 2, 'Banglore': 3, 'Kolkata': 4, 'Delhi': 5,
            'Kolkata': 1, 'Hyderabad': 2, 'Delhi': 3, 'Banglore': 4, 
            'Cochin': 5, 'New Delhi': 6}, inplace=True)


    out = model.predict(user_data)[0]

    st.subheader(f":orange[Predicted flight price: :blue[{out}]]")
    st.balloons()