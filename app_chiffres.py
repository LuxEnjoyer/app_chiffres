import json
import requests
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Définir une fonction pour créer et entraîner le modèle
def train_mnist_model():
    # Charger et préparer les données MNIST
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Créer le modèle
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

    # Entraîner le modèle
    batch_size = 256
    epochs = 1
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test))
    return model

def predict_digit(dataURL):
    # Préparez les données JSON à envoyer
    data = {
        "image": dataURL
    }
    
    # Définissez les en-têtes pour indiquer que vous envoyez des données JSON
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        # Effectuez une requête POST vers l'URL '/predict' avec les données JSON
        response = requests.post('/predict', data=json.dumps(data), headers=headers)
        
        # Analysez la réponse JSON
        response_data = response.json()
        predicted_digit = response_data.get("prediction")
        
        if predicted_digit is not None:
            print(f"Chiffre prédit : {predicted_digit}")
        else:
            print("Données de prédiction introuvables dans la réponse.")
    
    except Exception as error:
        print(f"Erreur : {error}")

# Interface graphique Streamlit
st.title('Application de reconnaissance de chiffres')

# Ajoutez la zone de saisie pour dessiner les chiffres
st.write("Dessinez un chiffre dans la zone ci-dessous:")
drawing = st_canvas(
    fill_color="black",
    stroke_width=10,
    stroke_color="white",
    background_color="#000000",
    height=200,
    width=200,
    drawing_mode="freedraw",
    key="canvas"
)

if st.button('Enregistrer l\'image'):
    if drawing.json_data:
        # Récupérer les données de dessin directement depuis drawing.json_data
        img_data = drawing.json_data
        # Créer une image à partir des données de dessin
        img = Image.fromarray(img_data)
        # Sauvegarder l'image
        img.save('dessin.png')
        st.success('Image enregistrée en tant que "dessin.png".')

# Bouton pour entraîner le modèle
if st.button('Entraîner le modèle'):
    trained_model = train_mnist_model()
    st.success('Entraînement terminé.')

# Vous pouvez ajouter d'autres éléments à votre interface graphique ici, par exemple, des widgets interactifs pour tester le modèle avec des images.

# Lancer l'application avec "streamlit run" dans le terminal