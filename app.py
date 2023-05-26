import numpy as np
import pandas as pd
import streamlit as st 
import io
from io import StringIO
from sklearn.model_selection import train_test_split
import joblib
import xlsxwriter
import os
import warnings


def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://hollywoodlife.com/wp-content/uploads/2018/06/donald-trump-rediculous-photos-ss-01.jpg?w=680");
             background-attachment: fixed;
             background-size: cover;
             text-color: white;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

def clean_column(df, column, patterns):
    for pattern, replacement in patterns.items():
        df[column] = df[column].str.replace(pattern, replacement)
        df[column] = df[column].str.lower() # applica il lower
    return df

def main():
    #add_bg_from_url()
    warnings.filterwarnings('ignore')
    uploaded_model = joblib.load('NLP-model.pkl')
    st.title("Analisi del sentiment")
    opzioni = {
        "Inserisci testo da analizzare da casella di testo":0,
        "Inserisci file di testo da analizzare":1
    }
    patterns = {
            r'\d+': ' ',          # remove digits (numeri)
            r'[\\]+[n]': '',      # questo rimuove \\n
            r'[^\w\s]': ' ',     # remove punteggiatura e simboli ...,'@!£$%
            r'\b\w{1,2}\b':'',      # remove all token less than2 characters
            r'(http|www)[^\s]+':'', # remove website
            r'\s+': ' ',            # sostituisce tutti i multipli spazi con uno spazio
            }

    to_do = st.radio('Scegli cosa fare', list(opzioni.keys()))

    valore_selezionato = opzioni[to_do]

    if valore_selezionato == 0:
        text = st.text_input("Inserisci il testo da analizzare","hello")

        if text != []:
            pred = uploaded_model.predict([text])
            st.write("Il sentiment della parola è:")
            if pred == 'positive':
                st.success(pred[0])
            else:
                st.error(pred[0])
    elif valore_selezionato == 1:
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            df=pd.read_csv(uploaded_file)
            clean_column(df, 'text', patterns)
            st.title("Panoramica sul dataframe")
            st.write(df)
            column = st.radio('Scegli la colonna dove si trova il testo da analizzare', list(df.columns))
            if ((df[column].astype(str).apply(lambda x: isinstance(x, str)).any() == True) | (df[column].isnull().any()==True)):
                
                pred = uploaded_model.predict(df[column])
                df2 = pd.DataFrame({'Predictions': pred})

                df_merged = pd.concat([df[column], df2], axis=1)
                st.write("Le predizioni ")
                st.write(df_merged)

                buffer = io.BytesIO()
                # download button 2 to download dataframe as xlsx
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    # Write each dataframe to a different worksheet.
                    df_merged.to_excel(writer, sheet_name='Sentiment', index=False)
                    # Close the Pandas Excel writer and output the Excel file to the buffer
                    writer.save()

                    download2 = st.download_button(
                        label="Download data as Excel",
                        data=buffer,
                        file_name='large_df.xlsx',
                        mime='application/vnd.ms-excel'
                    )
            else:
                st.error('La colonna non è valida per la sentiment analysis', icon="🚨")
        
if __name__=="__main__":
    main()