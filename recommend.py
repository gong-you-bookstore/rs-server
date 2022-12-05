from ast import literal_eval
import pandas as pd
import numpy as np
import warnings
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import random
warnings.filterwarnings(action='ignore')

books = pd.read_csv('./books_mod_fin.csv', encoding='utf-8')
books = books.replace({np.nan: 'none'})
books_df_nax = books.dropna(axis='rows')
tfidf = TfidfVectorizer(stop_words='english')


def find_sim_books(U):
    def userbooks(A):
        x = books_df_nax['isbn'].tolist()
        l = len(A)
        BD = []
        for i in range(l):
            if A[i] in x:
                BD.append(A[i])
        l2 = len(BD)
        n = random.randrange(0, l2)
        return BD[n]

    B = userbooks(U)

    def sim_idx_with_A(df, isbn):
        k = df[df['isbn'] == isbn]['kdc'].item()
        n = int(df[df['isbn'] == isbn]['des_cluster'])
        if n == 280:
            m = int(df[df['isbn'] == isbn]['tit_cluster'])
            same_clu_books_df = df[df['tit_cluster'] == m]
            sen = same_clu_books_df['content']
            kdc_same_vect = [0 for i in range(len(sen))]

            for i in range(len(sen)):
                if same_clu_books_df[i:i+1]['kdc'].item() == k:
                    kdc_same_vect[i] = 0.2

        else:
            same_clu_books_df = df[df['des_cluster'] == n]
            sen = same_clu_books_df['content']
            kdc_same_vect = [0 for i in range(len(sen))]

            for i in range(len(sen)):
                if same_clu_books_df[i:i+1]['kdc'].item() == k:
                    kdc_same_vect[i] = 0.05

        same_clu_books_idx = same_clu_books_df.loc[:, 'Unnamed: 0'].to_list()
        tit_embeddings = np.load(
            './tit_embeddings.npy')
        des_embeddings = np.load(
            './des_embeddings.npy')
        if n == 280:
            sen_embeddings = tit_embeddings[same_clu_books_idx, :]
        else:
            sen_embeddings = des_embeddings[same_clu_books_idx, :]

        clu_des_sim = cosine_similarity(sen_embeddings, sen_embeddings)
        idx = np.where(same_clu_books_df['isbn'] == B)[0][0]
        books_sim_vect = clu_des_sim[idx:idx+1]
        sim_vect_plus_w = books_sim_vect+kdc_same_vect
        books_des_sim_idx = sim_vect_plus_w.argsort()[::-1]

        return books_des_sim_idx

    sim_books_idx = sim_idx_with_A(books_df_nax, B)
    top_n = 20
    top_sim_idx = sim_books_idx[0][:top_n]
    top_sim_idx = top_sim_idx.reshape(-1,)
    sim_books = books_df_nax.iloc[top_sim_idx]
    outputs = sim_books[['isbn', 'title', 'author',
                         'content', 'thumbnail', 'publisher', 'kdc', 'price']]
    outputs_dic = outputs.to_dict('records')

    return outputs_dic
