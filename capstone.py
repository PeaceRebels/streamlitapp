#### Overall setup 
#########################################################

import streamlit as st
import pandas as pd
import waterfall_chart
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np
from PIL import Image
 


# Defining some general properties of the app
st.set_page_config(
    page_title= "Heart AI",
    page_icon = "🫀",
    layout="wide"
    )

# Define Load functions
@st.cache()
def load_data():
    data = pd.read_csv("200_patients.csv")
    return(data.dropna())


@st.cache(allow_output_mutation=True)
def load_model():
    filename = "SVM_classifier.sav"
    loaded_model = pickle.load(open(filename, "rb"))
    return(loaded_model)



# Load Data and Model
data = load_data()
model = load_model()

#Skalieren der kontinuierlichen Feature vor Prediction
def scaleData(df):
    from sklearn.preprocessing import StandardScaler
    relevant_columns = ['MentalHealth', 'BMI', 'PhysicalHealth', 'SleepTime']
    Scaler = StandardScaler()
    df_copy = df.copy()
    df_copy[relevant_columns] = Scaler.fit_transform(df_copy[relevant_columns])
    return df_copy


data2=data[data.columns].replace({'Yes':1, 'No':0, 'Male':1,'Female':0,'No, borderline diabetes':'0','Yes (during pregnancy)':'1' })
data2['Diabetic'] = data2['Diabetic'].astype(int)
  
data2 = pd.get_dummies(data2, drop_first = True)



# Test 
#fig, ax = plt.subplots()
#test = ax.hist(data2["BMI"])
#st.pyplot(test)


#### Define Header of app
#########################################################
row1_col1, row1_col2 = st.columns([4,1])

row1_col1.title("Willkommen bei HAI - Heart AI ❤️‍🩹")

row1_col1.markdown("Diese Web-App bietet Ihnen ein Tool zur Identifikation von Herzrisikopatienten")

image = Image.open("logo.jpg", )
row1_col2.image(image)


#### Definition of Alle Patienten 
#########################################################

st.header("Überblick aller Patienten")


# Model laufen lassen und Predictions machen 
predict = model.predict_proba(scaleData(data2))

predict = np.delete(predict, [0], 1)


predict = predict.round(2)



data2.insert(loc=0,
          column='Herzinfarktrisiko',
          value=predict) 



# 2 Spalten für Section 1 
row2_col1, row2_col2 = st.columns([1,1])

row2_col1.subheader("Visualisierung aller Patienten")




bubbly = plt.figure(figsize=(10, 4.5))
sns.histplot(x = "Herzinfarktrisiko", data=data2, color="#A32020")

#sns.scatterplot(data=data2, x="SleepTime", y="BMI", size="Herzinfarktrisiko", legend=False, sizes=(5, 800), color="#A32020")
row2_col1.pyplot(bubbly)


row2_col2.subheader("Top 10 Risikopatienten")



top10 = data2.sort_values('Herzinfarktrisiko', ascending=False)[["BMI",'Herzinfarktrisiko']].head(10)


names = ["Gesine Meta", "Helfried Oliver","Burkhard Evelyn", "Leberecht René","Heinz Alexandra", "Hiltrud Sabine","Konstanze Regula", "Otmar Ferdinand","Nico Engel", "Benedikt Eckehard"]
top10['BMI'] = names
top10=top10.rename(columns = {'BMI':'Name'})

top10.Herzinfarktrisiko = (top10.Herzinfarktrisiko * 100).astype(str) + '%'

    
row2_col2.table(top10)


st.write(" ")
st.write("------------ ")
st.write(" ")


#### Definition of Individuelles Patientenfile
#########################################################

st.header("Individuelle Diagnose")
uploaded_data = st.file_uploader("Laden Sie das Patienten-Profil hoch")


# Add action to be done if file is uploaded
if uploaded_data is not None:
    
    # Getting Data and Making Predictions
    new = pd.read_csv(uploaded_data)
    
    new = new[new.columns].replace({'Yes':1, 'No':0, 'Male':1,'Female':0,'No, borderline diabetes':'0','Yes (during pregnancy)':'1' })
    new['Diabetic'] = new['Diabetic'].astype(int)
    
    new = pd.get_dummies(new, drop_first = True)
        
    
   # Add User Feedback
    st.success("Sie haben erfolgreich ein neues Patienten-Profil hochgeladen!👍")
    
    row10_col1, row10_col2, row10_col3= st.columns([1,0.1,1])
  
    row10_col1.write(" ")
    row10_col1.write("Daten des aktuellen Patienten:")
    row10_col1.write(new[["BMI","Smoking","SleepTime","DiffWalking","Diabetic","PhysicalHealth"]])
    
    
    #st.write(new)
    proba = model.predict_proba(scaleData(new))[0][1]
    


    row10_col3.write(" ")
    
    
    proba = proba.round(4)
    proba = (proba * 100).astype(str) + '%'


    
    row10_col3.metric("Die Wahrscheinlichkeit eines Herzfinfarktes liegt bei:", proba)

    
st.write(" ")
st.write("------------ ")
st.write(" ")


#### Definition of Patient beraten
#########################################################

if uploaded_data is not None:


    st.header("Patient beraten")
    
    
    # Introducing Widget columns
    row4_col1, row4_col2, row4_col3, row4_col4 = st.columns([1,2,2,2])



    recommend = new.copy()


    row4_col1.subheader("Rauchen")
    smoke = row4_col1.radio(
        "Raucht der Patient?",
        ("Ja","Nein"),
        index=(0)
        )
    
    
    row4_col2.subheader("Physische Gesundheit")
    physical = row4_col2.slider("An wievielen Tagen hat der Patient physische Beschwerden gehabt?",
                     data2["PhysicalHealth"].min(),
                     data2["PhysicalHealth"].max(),
                     value=(10),
                     step=(1))
    
    
    
    row4_col3.subheader("Generelle Gesundheit")
    gen = row4_col3.select_slider("Wie schätzen sie die generelle Gesundheit des Patienten ein?",
                     options=['Poor', 'Fair', 'Good', 'Very good'],
                     value=("Fair"))
    
    
    
    row4_col4.subheader("Schlafzeit")
    sleep = row4_col4.slider("Wieviel Stunden pro Nacht schläft der Patient im Durchschnitt?",
                     data2["SleepTime"].min(),
                     data2["SleepTime"].max(),
                     value=(8))
    
    
    recommendStatic = recommend.copy()
    
    
    if smoke == "Ja":
        recommend["Smoking"] = 1
        recommendRauch = recommendStatic.copy()
        recommendRauch["Smoking"] = 1
    
    else:
        recommend["Smoking"] = 0
        recommendRauch = recommendStatic.copy()
        recommendRauch["Smoking"] = 0
    
    
    
    recommend["PhysicalHealth"] = physical
    recommendPhys = recommendStatic.copy()
    recommendPhys["PhysicalHealth"] = physical
    
    
    if gen == "Fair":
        recommend["GenHealth_Fair"] = 1
        recommend["GenHealth_Poor"] = 0
        recommend["GenHealth_Good"] = 0
        recommend["GenHealth_Very good"] = 0
        recommendGen = recommendStatic.copy()
        recommendGen["GenHealth_Fair"] = 1
        recommendGen["GenHealth_Poor"] = 0
        recommendGen["GenHealth_Good"] = 0
        recommendGen["GenHealth_Very good"] = 0
    
    elif gen == "Poor":
        recommend["GenHealth_Fair"] = 0
        recommend["GenHealth_Poor"] = 1
        recommend["GenHealth_Good"] = 0
        recommend["GenHealth_Very good"] = 0
        recommendGen = recommendStatic.copy()
        recommendGen["GenHealth_Fair"] = 0
        recommendGen["GenHealth_Poor"] = 1
        recommendGen["GenHealth_Good"] = 0
        recommendGen["GenHealth_Very good"] = 0
    
    elif gen == "Good":
        recommend["GenHealth_Fair"] = 0
        recommend["GenHealth_Poor"] = 0
        recommend["GenHealth_Good"] = 1
        recommend["GenHealth_Very good"] = 0
        recommendGen = recommendStatic.copy()
        recommendGen["GenHealth_Fair"] = 0
        recommendGen["GenHealth_Poor"] = 0
        recommendGen["GenHealth_Good"] = 1
        recommendGen["GenHealth_Very good"] = 0
    
    else:
        recommend["GenHealth_Fair"] = 0
        recommend["GenHealth_Poor"] = 0
        recommend["GenHealth_Good"] = 0
        recommend["GenHealth_Very good"] = 1
        recommendGen = recommendStatic.copy()
        recommendGen["GenHealth_Fair"] = 0
        recommendGen["GenHealth_Poor"] = 0
        recommendGen["GenHealth_Good"] = 0
        recommendGen["GenHealth_Very good"] = 1
    
    recommend["SleepTime"] = sleep
    recommendSleep = recommendStatic.copy()
    recommendSleep["SleepTime"] = sleep
    
    st.write(" ")
    st.write(" ")
   
    
    
    row5_col1, row5_col2 = st.columns([1,1])
    
    row5_col1.subheader("Aktuelle Werte")
    row5_col1.write(new[["Smoking", "PhysicalHealth", "GenHealth_Fair", "SleepTime"]])
    
    row5_col2.subheader("Angepasste Werte")
    row5_col2.write(recommend[["Smoking", "PhysicalHealth", "GenHealth_Fair", "SleepTime"]])
    
    
    
    
    a = ["Aktuell", "Rauchen", "Physische Gesundheit", "Gen. Gesundheit", "SleepTime"]
    b = []
    
    
    aktuell = model.predict_proba(scaleData(new))[0][1]
    

    
    
    b.append(aktuell)
    b.append(model.predict_proba(scaleData(recommendRauch))[0][1] - aktuell)
    b.append(model.predict_proba(scaleData(recommendPhys))[0][1]- aktuell)
    b.append(model.predict_proba(scaleData(recommendGen))[0][1]- aktuell)
    b.append(model.predict_proba(scaleData(recommendSleep))[0][1]- aktuell)
    
   
    
    aktuell = aktuell.round(3)

    newpercentage = model.predict_proba(scaleData(recommend))[0][1]
    
    newpercentage = round(newpercentage,2)
    newpercentage = (newpercentage * 100).astype(str) + '%'
    

    neuerwert = model.predict_proba(scaleData(recommend))[0][1] - aktuell
    neuerwert = round(neuerwert,2)
    neuerwert = (neuerwert * 100).astype(str) + '%'

    
    st.write(" ")
    st.write(" ")
    st.write(" ")


    row11_col1, row11_col2 = st.columns([1,2])


    row11_col1.subheader("Vergleich des Herzinfarktrisiko")

    row11_col1.metric("Die Wahrscheinlichkeit eines Herzinfarktes verändert sich wie folgt:", newpercentage, neuerwert)
    
    plt.rcParams["figure.figsize"] = (10,5)
    my_plot = waterfall_chart.plot(a,b,net_label="Neuer Wert", green_color="#A6A6A6", blue_color="#A32020", red_color = "#F5CACA")
    row11_col2.pyplot(my_plot)
    





































