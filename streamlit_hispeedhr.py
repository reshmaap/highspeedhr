# Import libraries
# Import necessary libraries for the code
import numpy as np
import pandas as pd
import re
import spacy
import pymysql
import ast

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline

import streamlit as st
from sqlalchemy import create_engine
import pickle, joblib

# Load the saved model
model = pickle.load(open('doc2vec_model.pkl', 'rb'))


#Load Saved Processed Resumes
resume_df = pd.read_csv(r"resume_processed.csv")
resume_df['text_vec'] = resume_df['text_vec'].apply(ast.literal_eval)

# Load the English language model in spaCy
nlp = spacy.load('en_core_web_sm')

def cleanRawText(rawText):
    rawText = str(rawText)
    rawText = re.sub('http\S+\s*', ' ', rawText)
    rawText = re.sub('RT|cc', ' ', rawText)
    rawText = re.sub('#\S+', '', rawText)
    rawText = re.sub('@\S+', '  ', rawText)
    rawText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', rawText)
    rawText = re.sub(r'[^\x00-\x7f]', r' ', rawText) 
    rawText = re.sub('\s+', ' ', rawText)
    rawText = re.sub('Job Description', '', rawText)
    return rawText

def remove_stop_words(text):
    if isinstance(text, str):
        doc = nlp(text)
        filtered_text = ' '.join([token.text for token in doc if not token.is_stop])
        return filtered_text
    else:
        return ''

def extract_entities(text):
    doc = nlp(text)
    named_entities = list(set([ent.text for ent in doc.ents]))
    return named_entities

def remove_words(text, words):
    pattern = r'\b(?:{})\b'.format('|'.join(map(re.escape, words)))
    return re.sub(pattern, '', text)

def cleanResumeData(df):
    df["Resume"] = df["Resume"].apply(lambda x: x.strip())
    cleaned_resume = df["Resume"].apply(cleanRawText)
    df["cleaned_text"] = cleaned_resume
    return df

def cleanJDData(df):
    df["Job_desc_raw"] = df["Job_desc_raw"].apply(lambda x: x.strip())
    cleaned_jd = df["Job_desc_raw"].apply(cleanRawText)
    df["cleaned_text"] = cleaned_jd
    return df


def predict(data, user, pw, db):
    engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")
    
    job_description_df = cleanJDData(data)
    # Apply the remove_stop_words function to each value in the 'cleaned_text' column
    job_description_df['cleaned_text_no_stopwords'] = job_description_df['cleaned_text'].apply(remove_stop_words)
    # Apply the extract_entities function to each value in the 'cleaned_text' column
    job_description_df['named_entities'] = job_description_df['cleaned_text'].apply(extract_entities)
    # Apply the remove_words function to each row in the job_description_df and resume_df
    job_description_df['cleaned_text_no_ne'] = job_description_df.apply(lambda row: remove_words(row['cleaned_text'], row['named_entities']), axis=1)
    
    #generate vectors
    job_description_text2vec = [model.infer_vector((job_description_df['cleaned_text_no_stopwords'][i].split(' '))) for i in range(0,len(job_description_df['cleaned_text_no_stopwords']))]
    
    job_description_text2vec_list = np.array(job_description_text2vec).tolist()
    job_description_df['text_vec'] = job_description_text2vec_list
    
    
    # Calculate cosine similarity between vectors in resume_df and job_description_df
    similarity_scores = cosine_similarity(job_description_df['text_vec'].tolist(), resume_df['text_vec'].tolist())

    # Create a new column in jd_raw_data_df with reference to top 5 cosine similarity scores
    top_5_scores = similarity_scores.argsort()[:, -5:][:, ::-1]  # Get the indices of top 5 scores for each row
    # Create a new column to store the top 5 similarity scores
    job_description_df['top_5_similarity_scores'] = [[similarity_scores[row][index] for index in indices] for row, indices in enumerate(top_5_scores)]
    # Create a new column to store the corresponding row indices in resume_df
    job_description_df['top_5_resume_indices'] = [resume_df.index[index_list].tolist() for index_list in top_5_scores]
    # Create a new column to store the corresponding categories in resume_df
    job_description_df['top_5_resume_category'] = [resume_df["Category"][index_list].tolist() for index_list in top_5_scores]
    
    # Save to table
    final = job_description_df[["Job_desc_raw","cleaned_text_no_stopwords","top_5_similarity_scores","top_5_resume_indices"]]

#     final.to_sql('job_resume_cos_scores', con = engine, if_exists = 'replace', chunksize = 1000, index = False)

    return final


def main():  

    st.title("Hi Speed Hr")
    st.sidebar.title("Upload Job Description Data")
    html_temp = """
    <div style="background-color:MediumSeaGreen;padding:8px">
    <h2 style="color:white;text-align:center;">Hr Assistant for proccessing resumes</h2>
    </div>
    
    """
    st.markdown(html_temp, unsafe_allow_html = True)
    st.text("")
    
    uploadedFile = st.sidebar.file_uploader("Choose a file", type = ['csv', 'xlsx'], accept_multiple_files = False, key = "fileUploader")
    if uploadedFile is not None :
        try:

            data = pd.read_csv(uploadedFile)
        except:
                try:
                    data = pd.read_excel(uploadedFile)
                except:      
                    data = pd.DataFrame(uploadedFile)
                
    else:
        st.sidebar.warning("You need to upload a csv or excel file.")
    
    html_temp = """
    <div style="background-color:MediumSeaGreen;padding:6px">
    <p style="color:white;text-align:center;">Add DataBase Credientials </p>
    </div>
    """
    st.sidebar.markdown(html_temp, unsafe_allow_html = True)
            
    user = st.sidebar.text_input("user", "Type Here")
    pw = st.sidebar.text_input("password", "Type Here")
    db = st.sidebar.text_input("database", "Type Here")
    
    result = ""
    
    if st.button("Resume Recommendations"):
        result = predict(data, user, pw, db)                           
        import seaborn as sns
        cm = sns.light_palette("blue", as_cmap = True)
        st.table(result.style.background_gradient(cmap = cm).set_precision(2))
                           
if __name__=='__main__':
    main()


