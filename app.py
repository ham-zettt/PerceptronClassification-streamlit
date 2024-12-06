import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report


def main():
    st.image("img/knn.png", width=100)

    with st.sidebar:
        page = option_menu("Pilih Halaman", [
                           "Home", "Data Understanding", "Preprocessing", "Model", "Evaluasi", "Testing"], default_index=0)

    if page == "Home":
        show_home()
    elif page == "Data Understanding":
        show_understanding()
    elif page == "Preprocessing":
        show_preprocessing()
    elif page == "Model":
        show_model()
    elif page == "Evaluasi":
        show_evaluasi()
    elif page == "Testing":
        show_testing()


def show_home():
    st.title(
        "Klasifikasi Penyakit Kanker Paru-Paru dengan menggunakan Metode K-Nearest Neighbors")

    # Explain what is Decision Tree
    st.header("Apa itu K-Nearest Neighbor?")
    st.write("K-Nearest Neighbor (KNN) merupakan salah satu algoritma yang digunakan untuk memprediksi kelas atau kategori dari data baru berdasarkan mayoritas kelas dari tetangga terdekat")

    # Explain the purpose of this website
    st.header("Tujuan Website")
    st.write("Website ini bertujuan untuk memberikan pemahaman mengenai tahapan proses pengolahan data dan klasifikasi dengan menggunakan metode KNN.")

    # Explain the data
    st.header("Data")
    st.write(
        "Data yang digunakan adalah Dataset Penyakit Kanker Paru-Paru diambil dari website Kaggle.")

    # Explain the process of Decision Tree
    st.header("Tahapan Proses Klasifikasi K-Nearest Neighbor")
    st.write("1. **Data Understanding atau Pemahaman Data**")
    st.write("2. **Preprocessing Data**")
    st.write("3. **Pemodelan**")
    st.write("4. **Evaluasi Model**")
    st.write("5. **Implementasi**")


def show_understanding():
    st.title("Data Understanding")
    data = pd.read_csv("survey lung cancer.csv")

    st.header("Metadata dari dataset Penyakit Kanker Paru-Paru")
    st.dataframe(data)

    col1, col2 = st.columns(2, vertical_alignment='top')

    with col1:
        st.write("Jumlah Data : ", len(data.axes[0]))
        st.write("Jumlah Atribut : ", len(data.axes[1]))

    with col2:
        st.write(
            f"Terdapat {len(data['LUNG_CANCER'].unique())} Label Kelas, yaitu : {data['LUNG_CANCER'].unique()}")

    st.markdown("---")

    st.header("Tipe Data & Missing Value")

    r2col1, r2col2 = st.columns(2, vertical_alignment='bottom')

    with r2col1:
        st.write("Tipe Data")
        st.write(data.dtypes)

    with r2col2:
        st.write("Missing Value")
        st.write(data.isnull().sum())

    st.markdown("---")

    st.header("Eksplorasi Data")

    st.dataframe(data.describe())

    st.markdown("---")

    target_counts = data['LUNG_CANCER'].value_counts()

    st.header('Distribusi Target')

    fig, ax = plt.subplots(figsize=(8, 6))
    target_counts.plot(kind='bar', color='blue', ax=ax)
    ax.set_title('Distribusi Target')
    ax.set_xlabel('Target')
    ax.set_ylabel('Jumlah')

    st.write(f"Jumlah Kelas YES : {target_counts['YES']}")
    st.write(f"Jumlah Kelas NO : {target_counts['NO']}")

    ax.set_xticks(range(len(target_counts.index)))
    ax.set_xticklabels(target_counts.index, rotation=0)

    st.pyplot(fig)

    st.markdown("---")

    cat_var = ['GENDER', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE',
               'CHRONIC DISEASE', 'FATIGUE', 'ALLERGY', 'WHEEZING', 'ALCOHOL CONSUMING',
               'COUGHING', 'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 'CHEST PAIN']

# Streamlit code to display the count plots
    st.header('Distribusi Fitur Kategorikal berdasarkan Target')

    for var in cat_var:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.countplot(x=var, hue='LUNG_CANCER', data=data, ax=ax)
        ax.set_title(f'Distribusi {var} berdasarkan Target')
        ax.set_xlabel(var)
        ax.set_ylabel('Jumlah')

        st.pyplot(fig)

    st.markdown("---")

    # Memilih fitur numerik untuk visualisasi korelasi
    numeric_data = data.select_dtypes(include=['number'])

    # Menghitung korelasi menggunakan metode Spearman
    all_features_corr = numeric_data.corr(method='spearman')

    # Definisikan fitur yang dipilih untuk korelasi
    specific_features = ['AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY',
                         'PEER_PRESSURE', 'CHRONIC DISEASE', 'FATIGUE', 'ALLERGY',
                         'WHEEZING', 'ALCOHOL CONSUMING', 'COUGHING', 'SHORTNESS OF BREATH',
                         'SWALLOWING DIFFICULTY', 'CHEST PAIN']

    # Menghitung korelasi untuk fitur yang dipilih
    specific_features = numeric_data[specific_features].corr(
        method='spearman')

    # Menampilkan Matriks Korelasi
    st.header('Correlation Matrices')

    # Matriks Korelasi untuk Semua Fitur
    st.subheader('Correlation Matrix untuk Semua Fitur Method Spearman')
    fig, ax = plt.subplots(figsize=(16, 12))
    sns.heatmap(all_features_corr, annot=True,
                fmt=".2f", cmap='coolwarm', ax=ax)
    ax.set_title('Correlation Matrix untuk Semua Fitur Method Spearman')
    st.pyplot(fig)
    plt.close(fig)


def show_preprocessing():
    st.title("Preprocessing")

    data = pd.read_csv("survey lung cancer.csv")

    fitur_columns = ['AGE', 'GENDER', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY',
                     'PEER_PRESSURE', 'CHRONIC DISEASE', 'FATIGUE', 'ALLERGY',
                     'WHEEZING', 'ALCOHOL CONSUMING', 'COUGHING', 'SHORTNESS OF BREATH',
                     'SWALLOWING DIFFICULTY', 'CHEST PAIN', 'LUNG_CANCER']

    data = data.drop(
        columns=[col for col in data.columns if col not in fitur_columns])

    st.header("Memilih Atribut yang digunakan untuk Pemodelan")
    st.dataframe(data)
    st.write("Jumlah Data : ", len(data.axes[0]))
    st.write("Jumlah Atribut : ", len(data.axes[1]))

    st.markdown("---")

    # Menampilkan bagian untuk mengecek dan menghapus data duplikat
    st.header("Mengechek dan Menghapus Data Duplikat")

    # Mengecek jumlah data duplikat
    duplikat_count = data.duplicated().sum()
    st.write("Jumlah Data Terduplikasi :", duplikat_count)

    # Menghapus data duplikat jika ada
    if duplikat_count > 0:
        data.drop_duplicates(inplace=True)
        st.write("Data setelah menghapus duplikat :")
        st.write("Jumlah Data :", len(data.axes[0]))

    st.markdown("---")

    # Menampilkan bagian untuk Label Encoding
    st.header("Label Encoding untuk Fitur Kategorikal")

    # Melakukan encoding pada fitur kategorikal menggunakan LabelEncoder
    label_encoder = LabelEncoder()
    columns_to_encode = ['GENDER', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE',
                         'CHRONIC DISEASE', 'FATIGUE', 'ALLERGY', 'WHEEZING', 'ALCOHOL CONSUMING',
                         'COUGHING', 'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 'CHEST PAIN']

    for col in columns_to_encode:
        if col in data.columns:
            data[col] = label_encoder.fit_transform(data[col])
            st.write(f"Fitur {col} setelah Label Encoding :")
            st.write(data[col].value_counts())
            st.write(f"Jumlah Kelas 0 : {data[col].value_counts().get(0, 0)}")
            st.write(f"Jumlah Kelas 1 : {data[col].value_counts().get(1, 0)}")

    # Menyimpan data setelah Label Encoding
    st.session_state['encoded_data'] = data

    st.markdown("---")

    # Normalisasi Data menggunakan Min-Max Scaler
    st.header("Normalisasi Data menggunakan Min Max Scalar")

    # Memisahkan fitur dan target
    x = data.drop(['LUNG_CANCER'], axis=1)
    y = data['LUNG_CANCER']

    # Normalisasi data
    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(x)

    x_scaled = pd.DataFrame(x_scaled, columns=x.columns)

    # Menampilkan data setelah normalisasi
    st.dataframe(x_scaled)

    # Menyimpan data yang telah dinormalisasi dalam session state
    st.session_state['preprocessed_data'] = x_scaled
    st.session_state['LUNG_CANCER'] = y


def show_model():
    st.title("Testing Model")

    # Pastikan data telah ada di session state sebelum menampilkan model
    if 'preprocessed_data' in st.session_state and 'LUNG_CANCER' in st.session_state:
        st.write("Data tersedia untuk model.")
        X_scaled = st.session_state['preprocessed_data']
        y = st.session_state['LUNG_CANCER']
        combined_data = pd.concat([X_scaled, y.reset_index(drop=True)], axis=1)
        st.dataframe(combined_data)

        st.markdown("---")

        st.header("Memecah menjadi data Training dan data Testing")

        # Memisahkan data training dan testing
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, random_state=0, train_size=0.8, shuffle=True)

        # Menampilkan data training
        trained = pd.concat([X_train, y_train], axis=1)
        st.write("### Data Training 80%")
        st.dataframe(trained)
        st.write("Jumlah Data : ", len(trained.axes[0]))

        # Menampilkan data testing
        testing = pd.concat([X_test, y_test], axis=1)
        st.write("### Data Testing 20%")
        st.dataframe(testing)
        st.write("Jumlah Data : ", len(testing.axes[0]))

        st.markdown("---")

        st.header("Testing menggunakan K = 7")

        # Melakukan pelatihan dengan KNN dengan K=7
        clf_KNN7 = KNeighborsClassifier(n_neighbors=7)
        clf_KNN7.fit(X_train, y_train)

        # Prediksi dan menampilkan hasil prediksi
        y_pred_KNN7 = clf_KNN7.predict(X_test)
        df_pred_KNN7 = pd.DataFrame(y_pred_KNN7, columns=["KNN7"])

        # Membandingkan hasil prediksi dengan kelas yang sebenarnya
        df_test = pd.DataFrame(y_test).reset_index(drop=True)

        df_pred_combined = pd.concat([df_pred_KNN7, df_test], axis=1)
        df_pred_combined.columns = ["KNN7", "Actual Class"]

        st.dataframe(df_pred_combined)
        st.session_state['y_pred'] = y_pred_KNN7
        st.session_state['y_test'] = y_test

    else:
        st.write(
            "### :red[Buka Menu Preprocessing terlebih dahulu jika halaman tidak menampilkan data]")


def show_evaluasi():
    st.title("Evaluasi Metode KNN")

    if 'y_pred' in st.session_state and 'y_test' in st.session_state:

        y_pred_KNN7 = st.session_state['y_pred']
        y_test = st.session_state['y_test']

        # Confusion Matrix untuk KNN
        unique_classes = y_test.unique()
        c_matrix_knn = confusion_matrix(
            y_test, y_pred_KNN7, labels=unique_classes)

        st.write("### Confusion Matrix (KNN)")
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(confusion_matrix=c_matrix_knn,
                               display_labels=unique_classes).plot(ax=ax)
        plt.title("Confusion Matrix for KNN")
        st.pyplot(fig)

        st.markdown("---")

        # Rumus Akurasi, Presisi, Recall, F1
        st.write("### Rumus Untuk menentukan Akurasi, Recall, Presisi, dan F1 Score")
        col1, col2 = st.columns(2)

        with col1:
            st.latex(r'Accuracy = \frac{TP + TN}{TP + TN + FP + FN}')
            st.latex(r'Recall = \frac{TP}{TP + FN}')

        with col2:
            st.latex(r'Precision = \frac{TP}{TP + FP}')
            st.latex(
                r'F1 Score = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}')

        st.markdown("---")

        # Evaluasi Metode KNN
        st.write("### Performance Metrics for KNN Model")
        accuracy_knn = accuracy_score(y_test, y_pred_KNN7) * 100
        precision_knn = precision_score(
            y_test, y_pred_KNN7, average='weighted') * 100
        recall_knn = recall_score(
            y_test, y_pred_KNN7, average='weighted') * 100
        f1_knn = f1_score(y_test, y_pred_KNN7, average='weighted') * 100

        col1, col2 = st.columns(2)
        with col1:
            st.write(f"#### Accuracy (KNN): {accuracy_knn:.2f}%")
            st.write(f"#### Recall (KNN): {recall_knn:.2f}%")
        with col2:
            st.write(f"#### Precision (KNN): {precision_knn:.2f}%")
            st.write(f"#### F1 Score (KNN): {f1_knn:.2f}%")

        st.markdown("---")

        # Memisahkan data training dan testing untuk Naive Bayes dan SVM
        X_train, X_test, y_train, y_test = train_test_split(
            st.session_state['preprocessed_data'], st.session_state['LUNG_CANCER'], random_state=0, train_size=0.8, shuffle=True)

        # Evaluasi Metode Naive Bayes
        st.title("Evaluasi Metode Naive Bayes")
        model_nb = GaussianNB()
        model_nb.fit(X_train, y_train)
        y_pred_nb = model_nb.predict(X_test)

        # Confusion Matrix untuk Naive Bayes
        c_matrix_nb = confusion_matrix(
            y_test, y_pred_nb, labels=unique_classes)
        st.write("### Confusion Matrix (Naive Bayes)")
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(confusion_matrix=c_matrix_nb,
                               display_labels=unique_classes).plot(ax=ax)
        plt.title("Confusion Matrix for Naive Bayes")
        st.pyplot(fig)

        # Evaluasi Metode Naive Bayes
        st.write("### Performance Metrics for Naive Bayes Model")
        accuracy_nb = accuracy_score(y_test, y_pred_nb) * 100
        precision_nb = precision_score(
            y_test, y_pred_nb, average='weighted') * 100
        recall_nb = recall_score(y_test, y_pred_nb, average='weighted') * 100
        f1_nb = f1_score(y_test, y_pred_nb, average='weighted') * 100

        col1, col2 = st.columns(2)
        with col1:
            st.write(f"#### Accuracy (Naive Bayes): {accuracy_nb:.2f}%")
            st.write(f"#### Recall (Naive Bayes): {recall_nb:.2f}%")
        with col2:
            st.write(f"#### Precision (Naive Bayes): {precision_nb:.2f}%")
            st.write(f"#### F1 Score (Naive Bayes): {f1_nb:.2f}%")

        st.markdown("---")

        # Evaluasi Metode SVM
        st.title("Evaluasi Metode SVM")
        model_svm = SVC(kernel='linear')
        model_svm.fit(X_train, y_train)
        y_pred_svm = model_svm.predict(X_test)

        # Confusion Matrix untuk SVM
        c_matrix_svm = confusion_matrix(
            y_test, y_pred_svm, labels=unique_classes)
        st.write("### Confusion Matrix (SVM)")
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(confusion_matrix=c_matrix_svm,
                               display_labels=unique_classes).plot(ax=ax)
        plt.title("Confusion Matrix for SVM")
        st.pyplot(fig)

        # Evaluasi Metode SVM
        st.write("### Performance Metrics for SVM Model")
        accuracy_svm = accuracy_score(y_test, y_pred_svm) * 100
        precision_svm = precision_score(
            y_test, y_pred_svm, average='weighted') * 100
        recall_svm = recall_score(y_test, y_pred_svm, average='weighted') * 100
        f1_svm = f1_score(y_test, y_pred_svm, average='weighted') * 100

        col1, col2 = st.columns(2)
        with col1:
            st.write(f"#### Accuracy (SVM): {accuracy_svm:.2f}%")
            st.write(f"#### Recall (SVM): {recall_svm:.2f}%")
        with col2:
            st.write(f"#### Precision (SVM): {precision_svm:.2f}%")
            st.write(f"#### F1 Score (SVM): {f1_svm:.2f}%")

        st.markdown("---")

    else:
        st.write(
            "### :red[Buka Menu Model terlebih dahulu jika halaman tidak menampilkan data]")


def show_testing():
    st.title("Testing Model")
    st.header("Lung Cancer Prediction")

    # Memuat model dan scaler
    with open("model/knn_pickle.pkl", "rb") as r:
        knnp = pickle.load(r)

    with open("model/scaler_pickle.pkl", "rb") as s:
        scaler = pickle.load(s)

    # Input data dari pengguna
    gender = st.selectbox(
        'Gender', [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
    age = st.number_input('Age', min_value=0, max_value=120, step=1)
    smoking = st.selectbox(
        'Smoking', [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    yellow_fingers = st.selectbox(
        'Yellow Fingers', [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    anxiety = st.selectbox(
        'Anxiety', [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    peer_pressure = st.selectbox(
        'Peer Pressure', [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    chronic_disease = st.selectbox(
        'Chronic Disease', [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    fatigue = st.selectbox(
        'Fatigue', [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    allergy = st.selectbox(
        'Allergy', [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    wheezing = st.selectbox(
        'Wheezing', [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    alcohol_consuming = st.selectbox(
        'Alcohol Consuming', [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    coughing = st.selectbox(
        'Coughing', [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    shortness_of_breath = st.selectbox(
        'Shortness of Breath', [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    swallowing_difficulty = st.selectbox('Swallowing Difficulty', [
                                         0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    chest_pain = st.selectbox(
        'Chest Pain', [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    if st.button('Predict'):
        # Menyiapkan data input
        input_data = [[gender, age, smoking, yellow_fingers, anxiety, peer_pressure,
                       chronic_disease, fatigue, allergy, wheezing, alcohol_consuming,
                       coughing, shortness_of_breath, swallowing_difficulty, chest_pain]]

        # Melakukan scaling pada data input
        input_data_scaled = scaler.transform(input_data)

        # Prediksi menggunakan model
        result = knnp.predict(input_data_scaled)

        # Menggunakan hasil prediksi langsung sebagai label
        prediction = result[0]  # Hasil prediksi sudah berupa 'YES' atau 'NO'

        # Menampilkan hasil prediksi
        st.subheader("Prediction Result:")
        st.write(f"Lung Cancer Prediction: :blue[{prediction}]")


if __name__ == "__main__":
    st.set_page_config(page_title="K-Nearest Neighbor Lung Cancer Prediction",
                       page_icon="img/knn.png")
    main()
