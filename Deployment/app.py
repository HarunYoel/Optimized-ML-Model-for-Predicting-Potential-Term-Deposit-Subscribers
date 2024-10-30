import streamlit as st
import pandas as pd
import joblib

st.header('Milestone 2 Model Deployment')
st.write("""
Created by Harun

Tujuan program ini adalah untuk memprediksi apakah seorang nasabah akan berlangganan deposito berjangka di sebuah Bank.
Dengan prediksi yang akurat, Bank dapat menargetkan nasabah yang paling potensial untuk mengikuti program deposito, sehingga meningkatkan efektivitas kampanye pemasaran.
""")
#load data
df = pd.read_csv('bank-additional-full.csv', sep=';')
st.write(df)

# buat sidebar untuk inputan user
st.sidebar.header("Fitur Input User")

def user_input():
    age = st.sidebar.number_input('Silahkan isi Umur Anda : ', value=35)
    job = st.sidebar.selectbox('Silahkan isi Perkerjaan Anda :', df['job'].unique())
    marital = st.sidebar.selectbox('Silahkan isi Status Kawin anda :', df['marital'].unique())
    education = st.sidebar.selectbox('Silahkan Isi Status Pendidikan Anda : ', df['education'].unique())
    default = st.sidebar.select_slider('Apakah anda memilki kartu kredit :', df['default'].unique())
    housing = st.sidebar.select_slider('Apakah anda memilki pinjaman rumah :', df['housing'].unique())
    loan = st.sidebar.select_slider('Apakah anda memiliki pinjaman pribadi : ', df['loan'].unique())
    contact = st.sidebar.select_slider('Jenis Teknologi Komunikasi yang digunakan :', df['contact'].unique())
    month = st.sidebar.selectbox('Bulan berapa kontak terakhir : ', df['month'].unique())
    day_of_week = st.sidebar.selectbox('Pada hari apa anda kontak :', df['day_of_week'].unique())
    duration = st.sidebar.number_input('Berapa lama anda melakukan kontak (detik) :', value=100)
    campaign = st.sidebar.number_input('Jumlah Campaign ketika ketika melakukan kontak :' , value=20)
    pdays = st.sidebar.number_input('Jumlah Hari berlalu setelah kontak dengan konsumen apabila belum pernah maka 999:', value=999)
    previous = st.sidebar.number_input('Jumlah Kampanya yang dilakukan sebelum kontak terakhir : ', value=0)
    poutcome = st.sidebar.selectbox('Hasil Status Kampanye : ', df['poutcome'].unique())
    emp_var_rate = st.sidebar.number_input('Tingkat Variasi Ketenagakerjaan - indikator kuartalan :', value = 1)
    cons_price_idx = st.sidebar.number_input('Indeks harga konsumen - indikator bulanan', value=0)
    cons_conft_idx = st.sidebar.number_input('Index Kepercayaan Konsumen : ', value=0)
    euribor3m = st.sidebar.number_input('Tingkat euribor 3 bulan - indikator harian : ', value=1)
    nr_employed = st.sidebar.number_input('Jumlah Karyawan - indikator bulanan :', value=0)

    data = {
            'age': age,
            'job': job,
            'marital': marital,
            'education': education,
            'default': default,
            'housing': housing,
            'loan': loan,
            'contact': contact,
            'month': month,
            'day_of_week': day_of_week,
            'duration': duration,
            'campaign': campaign,
            'pdays': pdays,
            'previous': previous,
            'poutcome': poutcome,
            'emp.var.rate': emp_var_rate,
            'cons.price.idx': cons_price_idx,
            'cons.conf.idx': cons_conft_idx,
            'euribor3m': euribor3m,
            'nr.employed': nr_employed
        }
    features = pd.DataFrame(data, index=[0])
    return features

input = user_input()
st.write(input)
#load data
pipe_cb = joblib.load('pipe_cb.pkl')
#data final
data_final = input.copy()
#predict
if st.button('predict'):
    prediciton = pipe_cb.predict(data_final)
    if prediciton == 'no':
        prediction = 'Tidak berlangganan Deposito'
    else : 
        prediction = 'Berlangganan Deposito'

    st.write('Berdasarkan user input, model dapat memprediksi sebagai berikut :')
    st.write(prediction)