# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 15:22:39 2024

@author: HP
"""

import numpy as np
import pickle
import streamlit as st

# Load the model and scaler
loaded_model = pickle.load(open('C:/Users/HP/Desktop/NEW ML/parkinson_model.sav', 'rb'))


def parkinson_predict(input_data):
    # Convert input data to numpy array
    input_data_as_numpy = np.asarray(input_data, dtype=float)

    # Reshape for model prediction
    input_data_reshaped = input_data_as_numpy.reshape(1, -1)

    # Standardize the data

    # Make prediction
    prediction = loaded_model.predict(input_data_reshaped)

    if prediction[0] == 1:
        return 'The person has Parkinson\'s disease'
    else:
        return 'The person does not have Parkinson\'s disease'

def main():
    st.title('Parkinson\'s Prediction Web App')

    # Input fields
    MDVP_Fo_Hz = st.text_input('MDVP:Fo(Hz)')
    MDVP_Fhi_Hz = st.text_input('MDVP:Fhi(Hz)')
    MDVP_Flo_Hz = st.text_input('MDVP:Flo(Hz)')
    MDVP_Jitter_percent = st.text_input('MDVP:Jitter(%)')
    MDVP_Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')
    MDVP_RAP = st.text_input('MDVP:RAP')
    MDVP_PPQ = st.text_input('MDVP:PPQ')
    Jitter_DDP = st.text_input('Jitter:DDP')
    MDVP_Shimmer = st.text_input('MDVP:Shimmer')
    MDVP_Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')
    Shimmer_APQ3 = st.text_input('Shimmer:APQ3')
    Shimmer_APQ5 = st.text_input('Shimmer:APQ5')
    MDVP_APQ = st.text_input('MDVP:APQ')
    Shimmer_DDA = st.text_input('Shimmer:DDA')
    NHR = st.text_input('NHR')
    HNR = st.text_input('HNR')
    RPDE = st.text_input('RPDE')
    DFA = st.text_input('DFA')
    spread1 = st.text_input('spread1')
    spread2 = st.text_input('spread2')
    D2 = st.text_input('D2')
    PPE = st.text_input('PPE')

    diagnosis = ''

    # Button for prediction
    if st.button('Parkinson\'s Test Results'):
        try:
            input_data = [
                MDVP_Fo_Hz, MDVP_Fhi_Hz, MDVP_Flo_Hz, MDVP_Jitter_percent,
                MDVP_Jitter_Abs, MDVP_RAP, MDVP_PPQ, Jitter_DDP,
                MDVP_Shimmer, MDVP_Shimmer_dB, Shimmer_APQ3, Shimmer_APQ5,
                MDVP_APQ, Shimmer_DDA, NHR, HNR, RPDE, DFA,
                spread1, spread2, D2, PPE
            ]
            input_data = [float(i) for i in input_data]  # Convert to float
            diagnosis = parkinson_predict(input_data)
        except ValueError:
            diagnosis = 'Please enter valid numerical values for all fields.'

    st.success(diagnosis)

if __name__ == '__main__':
    main()