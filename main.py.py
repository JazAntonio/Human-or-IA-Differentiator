# -*- coding: utf-8 -*-
"""IR_Proyecto.ipynb
"""

import pandas as pd
data_train = pd.read_csv("./drive/MyDrive/IR_proyecto/en_train.tsv", sep='\t')
data_test = pd.read_csv("./drive/MyDrive/IR_proyecto/en_test.tsv", sep='\t')
print(data_test.columns)

data_train.groupby(['domain']).count()

data_test.groupby(['domain']).count()

data_train = data_train.drop(["Unnamed: 0", "id", "prompt", "model", "domain"], axis=1)
data_test = data_test.drop(["Unnamed: 0", "id", "prompt", "model", "domain"], axis=1)

import re
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer


def docs_preproces(text):
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d', ' ', text)
    text = re.sub(r'_+', ' ', text)
    # text = re.sub(r'[^0-9]', ' ', text)

    text = text.lower()

    word_tokens = word_tokenize(text)

    # retrieve stem from words
    ps = PorterStemmer()
    stemming = []

    for w in word_tokens:
        stem = ps.stem(w)
        stemming.append(stem)

    wordnet_lemmatizer = WordNetLemmatizer()

    lemmatization = []
    for w in stemming:
        lemma = wordnet_lemmatizer.lemmatize(w)
        lemmatization.append(lemma)

    # Eliminar caracteres no latinos
    text_clean = ' '.join(lemmatization)
    text_clean = re.sub(r'[^\x00-\x7F]+', '', text_clean)
    text_clean = re.sub(r'([a-zA-Z])\1{2,}', '', text_clean)

    return text_clean

texts_for_train = list(data_train["text"])
texts_for_test = list(data_test["text"])
texts_train_preproced = [docs_preproces(text) for text in texts_for_train]
texts_test_preproced = [docs_preproces(text) for text in texts_for_test]

"""### Clasificacion con el vocabulario completo (sin stopwords), usando una SVM y una BoW con pesado TF-IDF"""

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer_tfidf = TfidfVectorizer(stop_words='english')
X_tfidf_train = vectorizer_tfidf.fit_transform(texts_train_preproced)
# Obtenemos el vocabulario
vocab_train = vectorizer_tfidf.get_feature_names_out()

vectorizer2_tfidf = TfidfVectorizer(vocabulary = vocab_train, stop_words='english')
X_tfidf_test = vectorizer2_tfidf.fit_transform(texts_test_preproced)

from sklearn.metrics import classification_report
from sklearn.svm import SVC

X_train = X_tfidf_train
X_test = X_tfidf_test
y_train = data_train['label']
y_test = data_test['label']

# Clasificación con SVM
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)
svm_pred = svm_classifier.predict(X_test)
print("Reporte de clasificación de SVM para BoW con pesado TF-IDF:\n\n", classification_report(y_test, svm_pred))

print("Usando todos los textos de la colección de entrenamiento, el vocabulario tiene un tamaño igual a:", len(vocab_train))

from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()
dtc.fit(X_train,y_train)
dct_prediction = dtc.predict(X_test)
print("Reporte de clasificación de DTC para BoW con pesado TF-IDF:\n\n", classification_report(y_test, dct_prediction))

from sklearn.linear_model import LogisticRegression

# Clasificación con regresión logística
logistic_classifier = LogisticRegression(max_iter=1000)
logistic_classifier.fit(X_train, y_train)
logistic_pred = logistic_classifier.predict(X_test)
print("Reporte de clasificación de regresión logística para BoW con pesado TF-IDF:\n\n", classification_report(y_test, logistic_pred))

data_train.groupby(["label"]).count().reset_index()

data_train_human = data_train[data_train["label"] == "human"]
data_train_gen = data_train[data_train["label"] == "generated"]

# Nos quedamos aproximadamente con el 20% de los datos humanos, realizando una seleccion aleatoria
human_sample_len = int(data_train_human.shape[0]/5)+1
data_train_human_sample = data_train_human.sample(n = human_sample_len)

# Nos quedamos aproximadamente con el 20% de los datos generados, realizando una seleccion aleatoria
gen_sample_len = int(data_train_gen.shape[0]/5)+1
data_train_gen_sample = data_train_gen.sample(n = gen_sample_len)

print(human_sample_len)
print(gen_sample_len)

data_train_human_sample.reset_index(drop = True, inplace = True)
data_train_gen_sample.reset_index(drop = True, inplace = True)

data_train_sample = pd.concat([data_train_human_sample, data_train_gen_sample])
data_train_sample

texts_sample_train = list(data_train_sample["text"])
texts_sample_train_preproced = [docs_preproces(text) for text in texts_sample_train]

texts_for_test = list(data_test["text"])
texts_test_preproced = [docs_preproces(text) for text in texts_for_test]

from sklearn.feature_extraction.text import CountVectorizer

vectorizer_count = CountVectorizer(stop_words='english')
X_train = vectorizer_count.fit_transform(texts_sample_train_preproced)
# Obtenemos el vocabulario
vocab_train = vectorizer_count.get_feature_names_out()

vectorizer2_count = CountVectorizer(vocabulary = vocab_train, stop_words='english')
X_test = vectorizer2_count.fit_transform(texts_test_preproced)

print("Usando un 20% los textos de la colección de entrenamiento, el vocabulario tiene un tamaño igual a:", len(vocab_train))

from sklearn.metrics import classification_report
from sklearn.svm import SVC

y_train = data_train_sample['label']
y_test = data_test['label']

# Clasificación con SVM
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)
svm_pred = svm_classifier.predict(X_test)
print("Reporte de clasificación de SVM para BoW con pesado TF:\n\n", classification_report(y_test, svm_pred))

from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()
dtc.fit(X_train,y_train)
dct_prediction = dtc.predict(X_test)
print("Reporte de clasificación de DTC para BoW con pesado TF:\n\n", classification_report(y_test, dct_prediction))

from sklearn.linear_model import LogisticRegression

# Clasificación con regresión logística
logistic_classifier = LogisticRegression(max_iter=1000)
logistic_classifier.fit(X_train, y_train)
logistic_pred = logistic_classifier.predict(X_test)
print("Reporte de clasificación de regresión logística para BoW con pesado TF-IDF:\n\n", classification_report(y_test, logistic_pred))

"""## Uso de embeddings"""

import numpy as np
from gensim.models import KeyedVectors

# Con la siguiente función se agregar el tamaño de vocabulario y la longitud de la representación,
# es decir, la longitud de los vectores de representación:
def add_word2vec_header(original_file, output_file):
    # Contar el tamaño del vocabulario y el tamaño del vector
    vocab_size = 0
    vector_size = 0

    with open(original_file, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) > 1:
                if vocab_size == 0:
                    vector_size = len(parts) - 1
                vocab_size += 1

    # Escribir el encabezado y el contenido original en el nuevo archivo
    with open(original_file, 'r', encoding='utf-8') as original, \
            open(output_file, 'w', encoding='utf-8') as output:
        output.write(f"{vocab_size} {vector_size}\n")
        for line in original:
            output.write(line)

# Ruta al archivo de texto que contiene a los embeddings pre-entrenados:
original_file = "./drive/MyDrive/IR_proyecto/glove6B50d.txt"
output_file = "./drive/MyDrive/IR_proyecto/salida.txt"

add_word2vec_header(original_file, output_file)  # Se agrega el encabezado correspondiente

#Se lee la representación vectorial de los "word embeddings" pre-entrenados:
word_vectors = KeyedVectors.load_word2vec_format(output_file, binary=False, encoding="utf8", unicode_errors='ignore')

train_embeddings = np.zeros((len(texts_sample_train_preproced), 50), dtype=object)

for id, text in enumerate(texts_sample_train_preproced):
    for word in text:
        try:
            temp_vector = word_vectors[word]
            train_embeddings[id][:] += temp_vector
        except:
            continue

test_embeddings = np.zeros((len(texts_test_preproced), 50), dtype=object)

for id, text in enumerate(texts_test_preproced):
    for word in text:
        try:
            temp_vector = word_vectors[word]
            test_embeddings[id][:] += temp_vector
        except:
            continue

from sklearn.metrics import classification_report
from sklearn.svm import SVC

y_train = data_train_sample['label']
y_test = data_test['label']

# División de datos
X_train = train_embeddings
X_test = test_embeddings

# Clasificación con SVM
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)
svm_pred2 = svm_classifier.predict(X_test)
print("Reporte de clasificación de SVM con el uso de embeddings de GloVe de 50 dim:\n\n", classification_report(y_test, svm_pred2))

from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()
dtc.fit(X_train,y_train)
dct_prediction = dtc.predict(X_test)
print("Reporte de clasificación de DTC con el uso de embeddings de GloVe de 50 dim:\n\n", classification_report(y_test, dct_prediction))

from sklearn.linear_model import LogisticRegression

# Clasificación con regresión logística
logistic_classifier = LogisticRegression(max_iter=1000)
logistic_classifier.fit(X_train, y_train)
logistic_pred = logistic_classifier.predict(X_test)
print("Reporte de clasificación de regresión logística con el uso de embeddings de GloVe de 50 dim:\n\n", classification_report(y_test, logistic_pred))

"""## Uso de BERT"""

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

# Cargar el modelo preentrenado BERT y el tokenizador
modelo_bert = BertModel.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Convertir textos a representaciones vectoriales con BERT
def obtener_vector_bert(texto):
    tokens = tokenizer.encode(texto, add_special_tokens=True)
    tokens_tensor = torch.tensor([tokens])
    with torch.no_grad():
        outputs = modelo_bert(tokens_tensor)
    return outputs[0].mean(dim=1).squeeze().numpy()

# Convertir los textos a vectores BERT
cls_bert_train = [obtener_vector_bert(texto) for texto in texts_sample_train_preproced]
cls_bert_test = [obtener_vector_bert(texto) for texto in texts_test_preproced]

from sklearn.metrics import classification_report
from sklearn.svm import SVC

y_train = data_train_sample['label']
y_test = data_test['label']

# División de datos
X_train = cls_bert_train
X_test = cls_bert_test

# Clasificación con SVM
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)
svm_pred = svm_classifier.predict(X_test)
print("Reporte de clasificación de SVM con el uso del CLS de BERT:\n\n", classification_report(y_test, svm_pred))

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

y_train = data_train_sample['label']
y_test = data_test['label']

# División de datos
X_train = cls_bert_train
X_test = cls_bert_test

dtc = DecisionTreeClassifier()
dtc.fit(X_train,y_train)
dct_prediction = dtc.predict(X_test)
print("Reporte de clasificación de DTC con el uso de BERT:\n\n", classification_report(y_test, dct_prediction))

from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

y_train = data_train_sample['label']
y_test = data_test['label']

# División de datos
X_train = cls_bert_train
X_test = cls_bert_test

# Clasificación con regresión logística
logistic_classifier = LogisticRegression(max_iter=1000)
logistic_classifier.fit(X_train, y_train)
logistic_pred = logistic_classifier.predict(X_test)
print("Reporte de clasificación de regresión logística con el uso de BERT:\n\n", classification_report(y_test, logistic_pred))

print(len(cls_bert_train[0]))

"""### Ahora se toman todas las instancias pero se disminuye la cantidad de dimensiones usando max features"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.svm import SVC

vectorizer_tfidf = TfidfVectorizer(max_features = 1000, stop_words='english')
X_tfidf_train = vectorizer_tfidf.fit_transform(texts_train_preproced)
# Obtenemos el vocabulario
vocab_train = vectorizer_tfidf.get_feature_names_out()

vectorizer2_tfidf = TfidfVectorizer(vocabulary = vocab_train, stop_words='english')
X_tfidf_test = vectorizer2_tfidf.fit_transform(texts_test_preproced)

X_train = X_tfidf_train
X_test = X_tfidf_test
y_train = data_train['label']
y_test = data_test['label']

# Clasificación con SVM
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)
svm_pred = svm_classifier.predict(X_test)
print("Reporte de clasificación de SVM para BoW con pesado TF-IDF con un vocabulario max de 1000 palabras:\n\n", classification_report(y_test, svm_pred))

from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()
dtc.fit(X_train,y_train)
dct_prediction = dtc.predict(X_test)
print("Reporte de clasificación de DTC para BoW con pesado TF-IDF:\n\n", classification_report(y_test, dct_prediction))

from sklearn.linear_model import LogisticRegression

# Clasificación con regresión logística
logistic_classifier = LogisticRegression(max_iter=1000)
logistic_classifier.fit(X_train, y_train)
logistic_pred = logistic_classifier.predict(X_test)
print("Reporte de clasificación de regresión logística para BoW con pesado TF-IDF:\n\n", classification_report(y_test, logistic_pred))

"""# Usando distribuciones de palabras"""

from collections import Counter
import numpy as np

# Función para calcular la distribución de palabras en un texto
def word_distribution(texto):
    palabras = texto.split()
    conteo_palabras = Counter(palabras)
    total_palabras = sum(conteo_palabras.values())
    distribucion = {palabra: frecuencia / total_palabras for palabra, frecuencia in conteo_palabras.items()}
    return distribucion

# Función para calcular la distancia de Jaccard entre dos distribuciones de palabras
def jaccard_distance(distribucion1, distribucion2):
    palabras_distr1 = set(distribucion1.keys())
    palabras_distr2 = set(distribucion2.keys())
    interseccion = palabras_distr1.intersection(palabras_distr2)
    union = palabras_distr1.union(palabras_distr2)
    distancia = len(interseccion) / len(union)
    return distancia


data_train_human = data_train[data_train["label"] == "human"]
data_train_gen = data_train[data_train["label"] == "generated"]

texts_human_train = list(data_train_human["text"])
texts_human_train_prepro = [docs_preproces(text) for text in texts_human_train]
human_train = ' '.join(texts_human_train_prepro)
human_train_distribution = word_distribution(human_train)

texts_gen_train = list(data_train_gen["text"])
texts_gen_train_prepro = [docs_preproces(text) for text in texts_gen_train]
gen_train = ' '.join(texts_gen_train_prepro)
gen_train_distribution = word_distribution(gen_train)

jaccard_distance(human_train_distribution, gen_train_distribution)

texts_for_test = list(data_test["text"])
texts_test_preproced = [docs_preproces(text) for text in texts_for_test]

jaccard_predict = []

for text in texts_test_preproced:
  text_distribution = word_distribution(text)
  if jaccard_distance(text_distribution, human_train_distribution) > jaccard_distance(text_distribution, gen_train_distribution):
    jaccard_predict.append("generated")
  else:
    jaccard_predict.append("human")

y_test = data_test['label']
print(classification_report(y_test, jaccard_predict))

from collections import Counter
import numpy as np

# Función para calcular la divergencia de Kullback-Leibler entre dos distribuciones de palabras
def kullback_leibler_divergency(distribucion1, distribucion2):
    palabras_comunes = set(distribucion1.keys()) & set(distribucion2.keys())
    kl_divergencia = 0
    for palabra in palabras_comunes:
        p = distribucion1[palabra]
        q = distribucion2[palabra]
        kl_divergencia += p * np.log(p / q)
    return kl_divergencia

kullback_leibler_divergency(human_train_distribution, gen_train_distribution)

kullback_leibler_predict = []

for text in texts_test_preproced:
  text_distribution = word_distribution(text)
  if kullback_leibler_divergency(text_distribution, human_train_distribution) > kullback_leibler_divergency(text_distribution, gen_train_distribution):
    kullback_leibler_predict.append("generated")
  else:
    kullback_leibler_predict.append("human")

y_test = data_test['label']
print(classification_report(y_test, kullback_leibler_predict))

import numpy as np
from collections import Counter
from scipy.stats import entropy

# Función para calcular la divergencia de Jensen-Shannon entre dos distribuciones de palabras
def jensen_shannon_divergency(distribucion1, distribucion2):
    # Obtener palabras comunes
    palabras_comunes = set(distribucion1.keys()) & set(distribucion2.keys())

    # Construir vectores de probabilidad
    p = np.array([distribucion1.get(palabra, 0) for palabra in palabras_comunes])
    q = np.array([distribucion2.get(palabra, 0) for palabra in palabras_comunes])

    # Calcular distribución media ponderada
    m = 0.5 * (p + q)

    # Calcular divergencia de Jensen-Shannon
    jsd = 0.5 * (entropy(p, m) + entropy(q, m))

    return jsd

jensen_shannon_divergency(human_train_distribution, gen_train_distribution)

jensen_shannon_predict = []

for text in texts_test_preproced:
  text_distribution = word_distribution(text)
  if jensen_shannon_divergency(text_distribution, human_train_distribution) > jensen_shannon_divergency(text_distribution, gen_train_distribution):
    jensen_shannon_predict.append("generated")
  else:
    jensen_shannon_predict.append("human")

y_test = data_test['label']
print(classification_report(y_test, jensen_shannon_predict))

"""## Aquí utilizamos la distribución de 11 emociones en texto"""

pip install nrclex

from collections import Counter
from nrclex import NRCLex

# Definir el texto de entrada
texto = "Im happy and angry"

# Crear un objeto NRCLex para el texto
lexico_human = NRCLex(human_train)
lexico_gen = NRCLex(gen_train)

# Obtener las emociones presentes en el texto y su frecuencia
emociones_human = lexico_human.affect_frequencies
emociones_gen = lexico_gen.affect_frequencies

# Imprimir la distribución de emociones
print("Distribución de emociones en el texto humano:")
print(emociones_human)

# Imprimir la distribución de emociones
print("Distribución de emociones en el texto generado:")
print(emociones_gen)

jaccard_distance(emociones_human, emociones_gen)

# kullback_leibler_divergency(emociones_human, emociones_gen)

jensen_shannon_divergency(emociones_human, emociones_gen)

js_emotions_predict = []

for text in texts_test_preproced:

  # Crear un objeto NRCLex para el texto
  lexico = NRCLex(text)

  # Obtener las emociones presentes en el texto y su frecuencia
  text_emotions = lexico.affect_frequencies

  if jensen_shannon_divergency(text_emotions, emociones_human) > jensen_shannon_divergency(text_emotions, emociones_gen):
    js_emotions_predict.append("generated")
  else:
    js_emotions_predict.append("human")

y_test = data_test['label']
print(classification_report(y_test, js_emotions_predict))