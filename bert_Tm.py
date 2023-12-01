import streamlit as st
from bertopic import BERTopic
import gensim
from gensim import corpora, models
import spacy
import re
from spacy.lang.en.stop_words import STOP_WORDS
from summarizer import Summarizer
from transformers import T5Tokenizer, T5ForConditionalGeneration

import pickle

bert_model_path = "/Users/juanita/TM_Bertopic/BERTopicmodel_50k_1.pkl"
loaded_bertopic_model = BERTopic.load(bert_model_path)

lda_model_path = "/Users/juanita/lda_model_project.pkl"

with open(lda_model_path, 'rb') as file:
    lda_model = pickle.load(file)

nlp_en = spacy.load('en_core_web_lg')

def preprocess_text(txt):
    txt = txt.lower()  
    txt = re.sub(r'[^a-zA-Z]', ' ', txt)
    docs = nlp_en(txt)
    word_list = [doc.lemma_ for doc in docs if doc.text not in STOP_WORDS]
    txt = " ".join(word_list)
    txt = txt.replace("-PRON-", "")
    txt = txt.replace("PRON", "")
    return txt


def perform_bert_topic_modeling(user_input):
        user_input_topic = loaded_bertopic_model.transform([user_input])[0]
        return user_input_topic
    
def get_topic_names(topic_indices):
  
    topic_names = [loaded_bertopic_model.get_topic(topic) for topic in topic_indices]
    return topic_names
    
def get_label(topic_indices):

    labels = [loaded_bertopic_model.custom_labels_[topic+1] for topic in topic_indices]
    return labels

def generate_summary(article_text):
    
    tokenizer = T5Tokenizer.from_pretrained("/Users/niharikabatra/Library/CloudStorage/GoogleDrive-niharikabatra111@gmail.com/My Drive/Niharika docs/UTA 2023 /Data Science/Project_Text/TokenizerT5")
    model = T5ForConditionalGeneration.from_pretrained("/Users/niharikabatra/Library/CloudStorage/GoogleDrive-niharikabatra111@gmail.com/My Drive/Niharika docs/UTA 2023 /Data Science/Project_Text/SummarizerT5")

    inputs = tokenizer.encode("summarize: " + article_text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs, max_length=150, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary
    

def topic_modeling_app():
    st.title('Topic Modeling App')

    article_text = st.text_area('Enter your article:', height=200)
    user_input = preprocess_text(article_text)

    user_input_topic = perform_bert_topic_modeling(user_input)
    user_input_topic_names = get_topic_names(user_input_topic)
    user_input_topic_label = get_label(user_input_topic)
  
    st.subheader(f'This article belongs to Topic {user_input_topic}: {user_input_topic_label}')
    st.subheader(f'Some of the keywords are : {user_input_topic_names}')

    summary = generate_summary(article_text)
    st.subheader('Summary:')
    st.write(summary)

if __name__ == '__main__':
    topic_modeling_app()
