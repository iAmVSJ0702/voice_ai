import streamlit as st
from audio_recorder_streamlit import audio_recorder
from groq import Groq
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()
client = Groq(api_key=os.getenv('GROQ_API_KEY'))
model = 'whisper-large-v3'

#Front end using streamlit
def frontend():
    st.title("Voice AI Demo")
    status_placeholder = st.empty()
    status_placeholder.write("Press Mic button to start asking question")
    recorded_audio = audio_recorder()
    if recorded_audio:
        status_placeholder.write("Converting audio ...")
        data_to_file(recorded_audio)
        status_placeholder.write("Audio conversion done.")
        status_placeholder.write("Convering audio to text and making transcription...")
        transcription = audio_to_text("temp_audio.wav")
        status_placeholder.write("Transcription is now made.")
        status_placeholder.write("Getting response...")
        response = answer(transcription)
        status_placeholder.write("Press mic button again to ask more questions")
        st.write("Q:" + transcription)
        st.write("A: " + response)

#Fuction to convert audio data to audio file
def data_to_file(recorded_audio):
    temp_audio_path = "temp_audio.wav"
    with open(temp_audio_path, "wb") as temp_file:
        temp_file.write(recorded_audio)


#Function for audio to text
def audio_to_text(audio_path):
    with open(audio_path, 'rb') as file:
        transcription = client.audio.translations.create(
            file=(audio_path, file.read()),
            model='whisper-large-v3',
        )
    return transcription.text

#Function for answerig User Query
def answer(user_question):
    model = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.6
    )

    prompt = ChatPromptTemplate([
        ("system", "You are super knowlegable AI chat bot whuch will answer all User Query, answer with confident, also this response will get convert back to speech, so dont make point or anything, but make your answer in para form and dont make it too large, and use proper annotation, comma, full stop, question mark, so that a better text to speach can be genrate back."),
        ("user", "User Query: {question}"),
    ])

    parser = StrOutputParser()

    chain = prompt|model|parser
    answer = chain.invoke({'question': user_question})
    return answer

frontend()