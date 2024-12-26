import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu

from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.feature_selection import mutual_info_classif

from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

from collections import Counter
from imblearn.over_sampling import SMOTE



def main():
    st.image("img/nn.jpg", width=100)

    with st.sidebar:
        page = option_menu("Pilih Halaman", [
                           "Home", "Data Understanding", "Preprocessing", "Seleksi Fitur", "Model", "Evaluasi", "Testing"], default_index=0)

    if page == "Home":
        show_home()
    elif page == "Data Understanding":
        show_understanding()
    elif page == "Preprocessing":
        show_preprocessing()
    elif page == "Seleksi Fitur":
        seleksi_fitur()
    elif page == "Model":
        show_model()
    elif page == "Evaluasi":
        show_evaluasi()
    elif page == "Testing":
        show_testing()

def show_home():
    st.title(
        "Klasifikasi Penyakit Ginjal Kronis Menggunakan Metode Perceptron")

    # Explain what is Perceptron
    st.header("Apa itu Perceptron?")
    st.write("Perceptron adalah algoritma pembelajaran mesin yang digunakan untuk klasifikasi biner. Algoritma ini bekerja dengan cara menghitung bobot pada setiap fitur dan membuat keputusan berdasarkan hasil perhitungan tersebut.")

    # Explain the purpose of this website
    st.header("Tujuan Website")
    st.write("Website ini bertujuan untuk memberikan pemahaman mengenai tahapan proses pengolahan data dan klasifikasi dengan menggunakan metode Perceptron.")

    # Explain the data
    st.header("Data")
    st.write(
        "Data yang digunakan adalah Dataset Penyakit Ginjal Kronis yang bertujuan untuk mengklasifikasikan apakah seseorang menderita penyakit ginjal kronis atau tidak.")

    # Explain the process of Decision Tree
    st.header("Tahapan Proses Klasifikasi K-Nearest Neighbor")
    st.write("1. **Data Understanding atau Pemahaman Data**")
    st.write("2. **Preprocessing Data**")
    st.write("3. **Pemodelan**")
    st.write("4. **Evaluasi Model**")
    st.write("5. **Implementasi**")


def show_understanding():
    st.title("Data Understanding")
    data = pd.read_csv("kidney_disease.csv")  # Pastikan file CSV sesuai dengan nama yang diunduh

    st.header("Metadata dari dataset Penyakit Ginjal Kronis")
    st.dataframe(data)

    col1, col2 = st.columns(2, vertical_alignment='top')

    with col1:
        st.write("Jumlah Data : ", len(data.axes[0]))
        st.write("Jumlah Atribut : ", len(data.axes[1]))

    with col2:
        st.write(
            f"Terdapat {len(data['classification'].unique())} Label Kelas, yaitu : {data['classification'].unique()}")

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

    # Distribusi Target
    target_counts = data['classification'].value_counts()

    st.header('Distribusi Target')

    fig, ax = plt.subplots(figsize=(8, 6))
    target_counts.plot(kind='bar', color='blue', ax=ax)
    ax.set_title('Distribusi Target')
    ax.set_xlabel('Target')
    ax.set_ylabel('Jumlah')

    st.write(f"Jumlah Kelas CKD (Chronic Kidney Disease) : {target_counts['ckd']}")
    st.write(f"Jumlah Kelas NOT CKD (Non-Chronic Kidney Disease) : {target_counts['notckd']}")

    ax.set_xticks(range(len(target_counts.index)))
    ax.set_xticklabels(target_counts.index, rotation=0)

    st.pyplot(fig)

    st.markdown("---")

    # Fitur kategorikal yang relevan berdasarkan dataset
    cat_var = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'sod', 
               'pot', 'hemo', 'pcv', 'wc', 'rc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']

    st.header('Distribusi Fitur Kategorikal berdasarkan Target')

    for var in cat_var:
        if var != 'classification':  # Skip the target variable
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.countplot(x=var, hue='classification', data=data, ax=ax)
            ax.set_title(f'Distribusi {var} berdasarkan Target')
            ax.set_xlabel(var)
            ax.set_ylabel('Jumlah')

            st.pyplot(fig)

    st.markdown("---")

    # Memilih fitur numerik untuk visualisasi korelasi
    numeric_data = data.select_dtypes(include=['number'])

    # Menghitung korelasi menggunakan metode Spearman
    all_features_corr = numeric_data.corr(method='spearman')

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

    # Load the dataset
    data = pd.read_csv("kidney_disease.csv")  # Ganti dengan nama file dataset yang sesuai

    # Drop unnecessary columns
    data = data.drop(columns=['id'], errors='ignore')  # Jika 'id' ada, akan dihapus

    # Set the target column
    target_column = 'classification'
    data[target_column] = data[target_column].str.strip()  # Hapus spasi ekstra pada kolom target

    st.header("Memilih Atribut yang digunakan untuk Pemodelan")
    st.dataframe(data)
    st.write("Jumlah Data : ", len(data.axes[0]))
    st.write("Jumlah Atribut : ", len(data.axes[1]))

    st.markdown("---")

    # Transformasi Data - Label Encoding untuk fitur kategorikal
    data_categorical = data.drop(columns=[target_column]).select_dtypes(include=['object']).columns
    label_encoders = {}
    for col in data_categorical:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        label_encoders[col] = le

    st.header("Label Encoding untuk Fitur Kategorikal")
    for col in data_categorical:
        st.write(f"Fitur {col} setelah Label Encoding :")
        st.write(data[col].value_counts())
        st.write(f"Jumlah Kelas 0 : {data[col].value_counts().get(0, 0)}")
        st.write(f"Jumlah Kelas 1 : {data[col].value_counts().get(1, 0)}")

    st.markdown("---")

    # Mengatasi Missing Values
    st.header("Mengatasi Missing Values")

    imputer_mean = SimpleImputer(strategy='mean')  # Untuk kolom numerik
    imputer_mode = SimpleImputer(strategy='most_frequent')  # Untuk kolom kategorikal

    # Kolom numerik
    data_numeric = data.select_dtypes(include=['float64', 'int64']).columns
    data[data_numeric] = imputer_mean.fit_transform(data[data_numeric])

    # Kolom kategorikal
    data_categorical = data.select_dtypes(include=['object']).columns
    data[data_categorical] = imputer_mode.fit_transform(data[data_categorical])

    st.write("Data setelah mengatasi missing values:")
    st.dataframe(data)

    st.markdown("---")

    # Normalisasi Data menggunakan Min-Max Scaler
    st.header("Normalisasi Data menggunakan Min Max Scalar")

    scaler = MinMaxScaler()
    data[data_numeric] = scaler.fit_transform(data[data_numeric])

    st.write("Data setelah normalisasi:")
    st.dataframe(data)

    # Menyimpan data yang telah diproses dalam session state
    st.session_state['preprocessed_data'] = data
    st.session_state['classification'] = data[target_column]


def seleksi_fitur():
    if 'preprocessed_data' in st.session_state and 'classification' in st.session_state:
        st.title("Seleksi Fitur Berdasarkan Information Gain dan Pembalancingan Data")

        # Ambil data yang sudah diproses dan target dari session state
        data = st.session_state['preprocessed_data']
        target_column = 'classification'
        y = st.session_state['classification']
        
        # Memisahkan fitur dan target
        X = data
        X[target_column] = y

        # Menghitung Information Gain
        information_gain = mutual_info_classif(X.drop(columns=[target_column]), y, discrete_features='auto')
        
        # Membuat DataFrame untuk menampilkan Information Gain
        information_gain_df = pd.DataFrame({
            'Feature': X.drop(columns=[target_column]).columns,
            'Information Gain': information_gain
        }).sort_values(by='Information Gain', ascending=False)

        # Menampilkan Information Gain setiap fitur
        st.header("Information Gain setiap fitur:")
        st.dataframe(information_gain_df)

        # Seleksi fitur yang memiliki Information Gain lebih besar dari threshold (0.2)
        threshold = 0.2
        selected_features = information_gain_df[information_gain_df['Information Gain'] > threshold]['Feature']
        
        st.header(f"Fitur yang dipilih (Information Gain > {threshold}):")
        st.write(selected_features)

        # Memilih data dengan fitur yang telah dipilih
        X_selected = X[selected_features]

        # Menambahkan kolom target kembali
        data_modified = pd.concat([X_selected, y], axis=1)

        # Melakukan Label Encoding pada kolom target
        data_modified[target_column] = LabelEncoder().fit_transform(data_modified[target_column])

        # Tampilkan data yang sudah dimodifikasi
        st.header("Data yang Dimodifikasi dengan Fitur Terpilih:")

        # Menyimpan data yang sudah dimodifikasi di session state
        st.session_state['data_with_selected_features'] = data_modified

        X_resampled, y_resampled = SMOTE(random_state=42).fit_resample(X_selected, y)

        # Menggabungkan data yang telah dibalancing
        balanced_data = pd.concat([pd.DataFrame(X_resampled, columns=X_selected.columns), pd.DataFrame(y_resampled, columns=[target_column])], axis=1)

        # Tampilkan distribusi kelas setelah balancing
        balanced_target_counts = Counter(balanced_data[target_column])
        st.write(balanced_target_counts)

        # Tampilkan data yang telah dibalancing
        st.dataframe(balanced_data.head())

        # Menyimpan data yang telah dibalancing di session state
        st.session_state['balanced_data'] = data_modified

    else:
        st.write(":red[Pastikan bahwa data telah diproses terlebih dahulu di menu Preprocessing.]")


def show_model():
    st.title("Testing Model")

    # Pastikan data telah ada di session state sebelum menampilkan model
    if 'balanced_data' in st.session_state:
        balanced_data = st.session_state['balanced_data']
        st.write("Data tersedia untuk model.")
        X = balanced_data.drop(columns=['classification'])
        y = balanced_data['classification']
        
        combined_data = pd.concat([X, y.reset_index(drop=True)], axis=1)
        st.dataframe(combined_data)

        st.markdown("---")

        st.header("Memecah menjadi data Training dan data Testing")

        # Memisahkan data training dan testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=0.8, shuffle=True)

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
    
    else:
        st.write(":red[Pastikan bahwa data telah diproses sebelumnya.]")


# Fungsi untuk menampilkan confusion matrix
def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    st.pyplot(fig)

def train_evaluate(X_train, X_test, y_train, y_test, model_type='perceptron'):
    # Evaluasi model dan tampilkan confusion matrix
    if model_type == 'perceptron':
        model = Perceptron(max_iter=50, tol=1e-3)
        model.fit(X_train, y_train)
        y_pred = (model.predict(X_test) > 0.5).astype(int)
    
    elif model_type == 'naive_bayes':
        model = GaussianNB()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    elif model_type == 'logistic_regression':
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    # Evaluasi akurasi
    accuracy = accuracy_score(y_test, y_pred)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    return accuracy, cm

def show_evaluasi():
    st.title("Evaluasi Model")

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

    # Pastikan data telah ada di session state sebelum menampilkan model
    if 'balanced_data' in st.session_state:
        data = st.session_state['balanced_data']
        X = data.drop(columns=['classification'])
        y = data['classification']

        # Memisahkan data training dan testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=0.8, shuffle=True)

        # Menampilkan data training dan testing
        st.write("### Data Training dan Testing")
        st.write("Jumlah Data Training: ", len(X_train))
        st.write("Jumlah Data Testing: ", len(X_test))

        # Model evaluation
        models = ['perceptron', 'naive_bayes', 'logistic_regression']
        
        for model_type in models:
            st.write(f"### Evaluasi Model: {model_type.capitalize()}")

            # Evaluasi model
            accuracy, cm = train_evaluate(X_train, X_test, y_train, y_test, model_type=model_type)
            st.write(f"Akurasi Model {model_type.capitalize()}: {accuracy:.4f}")

            # Menampilkan confusion matrix
            plot_confusion_matrix(cm, class_names=["Negative", "Positive"])

            st.markdown("---")

    else:
        st.write(":red[Pastikan bahwa data telah diproses sebelumnya.]")




def show_testing():
    st.title("Testing Model")
    st.header("Chronic Kidney Disease Prediction")

    # Memuat model dan scaler
    with open("model/perceptron.pkl", "rb") as r:
        perceptron_pickle = pickle.load(r)

    with open("model/scaler.pkl", "rb") as s:
        scaler_pickel = pickle.load(s)

    # Input data dari pengguna
    hemo = st.number_input('Hemoglobin (hemo)', min_value=0.0, step=0.1)
    pcv = st.number_input('Packed Cell Volume (pcv)', min_value=0, max_value=100, step=1)
    rc = st.number_input('Red Blood Cells (rc)', min_value=0, max_value=10, step=1)
    sg = st.number_input('Specific Gravity (sg)', min_value=1.00, max_value=1.50, step=0.01)
    sc = st.number_input('Sodium Content (sc)', min_value=0, max_value=150, step=1)
    al = st.number_input('Albumin (al)', min_value=1, max_value=5, step=1)
    rbc = st.number_input('Red Blood Cell Count (rbc)', min_value=0, max_value=10, step=1)
    sod = st.number_input('Sodium (sod)', min_value=0, max_value=200, step=1)
    htn = st.selectbox('Hypertension (htn)', [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    pot = st.number_input('Potassium (pot)', min_value=0, max_value=10, step=1)
    dm = st.selectbox('Diabetes Mellitus (dm)', [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")  
    
    # Tombol untuk memulai prediksi
    if st.button('Predict'):
        # Menyiapkan data input
        input_data = [[hemo, pcv, rc, sg, sc, al, rbc, sod, htn, pot, dm]]

        # Melakukan scaling pada data input
        input_data_scaled = scaler_pickel.transform(input_data)

        # Prediksi menggunakan model
        result = perceptron_pickle.predict(input_data_scaled)

        # Menggunakan hasil prediksi untuk menentukan label
        prediction = "Positive" if result[0] == 1 else "Negative"

        # Menampilkan hasil prediksi
        st.subheader("Hasil Prediksi: ")
        st.write(f"Prediksi Penyakit Ginjal Kronis: :blue[{prediction}]")

if __name__ == "__main__":
    st.set_page_config(page_title="Klasifikasi Penyakit Ginjal Kronis Dengan Metode Perceptron",
                       page_icon="img/nn.jpg")
    main()
