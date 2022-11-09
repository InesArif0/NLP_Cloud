import streamlit as st
from predictions import predict

st.title("Topics Analyze")
text = st.text_input('Critique à analyser')

nbrTopics = st.number_input('Nombre de topics')
bool_button = st.button('Détecter le topic')

if bool_button == True :
    polarity, topics = predict(text, int(nbrTopics))
    if polarity > 0:
        st.warning("Polarité: " + str(polarity) + " (COMMENTAIRE POSITIF)")
    else :
        st.warning("Polarité: " + str(polarity) + " (COMMENTAIRE NEGATIF)")
        st.info("Topics :")
        st.info(topics)




