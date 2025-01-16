import os
import numpy as np
import tensorflow as tf
import cv2
import streamlit as st
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.applications import MobileNet
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# Função de carregamento de dados
def load_data(uploaded_files):
    empty_files = [file for file in uploaded_files if 'empty' in file.name]
    not_empty_files = [file for file in uploaded_files if 'not_empty' in file.name]
    
    print(f"Number of empty images: {len(empty_files)}")
    print(f"Number of not_empty images: {len(not_empty_files)}")
    
    return empty_files, not_empty_files

# Função para criar generadores de dados a partir de arquivos carregados
def create_image_generators_from_uploaded(uploaded_files, batch_size=32, img_size=(150, 150)):
    # Use ImageDataGenerator para carregar as imagens carregadas
    datagen = ImageDataGenerator(rescale=1./255)
    
    # Para facilitar, dividimos as imagens carregadas em diretórios temporários
    temp_dir = "/tmp/streamlit_images"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Criar diretórios para 'empty' e 'not_empty' com base no nome dos arquivos
    empty_dir = os.path.join(temp_dir, 'empty')
    not_empty_dir = os.path.join(temp_dir, 'not_empty')
    os.makedirs(empty_dir, exist_ok=True)
    os.makedirs(not_empty_dir, exist_ok=True)

    # Salve as imagens nos respectivos diretórios
    for file in uploaded_files:
        if 'empty' in file.name:
            file_path = os.path.join(empty_dir, file.name)
        else:
            file_path = os.path.join(not_empty_dir, file.name)
        
        with open(file_path, "wb") as f:
            f.write(file.getvalue())

    # Criar generadores de dados a partir dos diretórios temporários
    train_datagen = datagen.flow_from_directory(
        temp_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='training'
    )

    validation_datagen = datagen.flow_from_directory(
        temp_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation'
    )

    return train_datagen, validation_datagen

# Função para construir e treinar um modelo CNN
def train_cnn_model(train_generator, validation_generator, epochs=10):
    CNN_model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')  # Camada de saída com ativação sigmoide para binário
    ])

    CNN_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = CNN_model.fit(train_generator, epochs=epochs, validation_data=validation_generator)

    return CNN_model, history

# Função para criar o modelo com Transfer Learning usando MobileNet
def train_mobilenet_model(train_generator, validation_generator, epochs=10):
    base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator
    )

    return model, history

# Função de avaliação
def evaluate_model(model, validation_generator):
    val_loss, val_acc = model.evaluate(validation_generator)
    st.write(f'Validation Loss: {val_loss}')
    st.write(f'Validation Accuracy: {val_acc}')
    
    # Gerar as previsões
    predictions = model.predict(validation_generator, verbose=1)
    predictions = (predictions > 0.5).astype(int)

    # Relatório de classificação
    y_true = validation_generator.classes
    report = classification_report(y_true, predictions, target_names=validation_generator.class_indices)
    st.text(report)

    # Matriz de confusão
    conf_matrix = confusion_matrix(y_true, predictions)
    disp = ConfusionMatrixDisplay(conf_matrix, display_labels=validation_generator.class_indices)
    disp.plot(cmap='Blues')
    st.pyplot()

# Função principal para a interface do Streamlit
def main():
    st.title("Parking Lot Occupancy Detection")

    # Upload de arquivos
    uploaded_files = st.file_uploader("Carregue as imagens do dataset", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
        # Carregar os dados
        empty_files, not_empty_files = load_data(uploaded_files)

        # Criar generadores de dados
        train_generator, validation_generator = create_image_generators_from_uploaded(uploaded_files)

        # Treinar o modelo CNN
        st.subheader("Treinando o modelo CNN")
        cnn_model, cnn_history = train_cnn_model(train_generator, validation_generator)

        # Mostrar o histórico de treinamento
        st.subheader("Gráfico de Acurácia e Perda")
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        axes[0].plot(cnn_history.history['accuracy'], label='Acurácia de Treinamento')
        axes[0].plot(cnn_history.history['val_accuracy'], label='Acurácia de Validação')
        axes[0].legend(loc='lower right')
        axes[0].set_title('Acurácia')

        axes[1].plot(cnn_history.history['loss'], label='Perda de Treinamento')
        axes[1].plot(cnn_history.history['val_loss'], label='Perda de Validação')
        axes[1].legend(loc='upper right')
        axes[1].set_title('Perda')

        st.pyplot(fig)

        # Avaliar o modelo
        st.subheader("Avaliação do Modelo CNN")
        evaluate_model(cnn_model, validation_generator)

        # Treinar o modelo MobileNet
        st.subheader("Treinando o modelo MobileNet")
        mobilenet_model, mobilenet_history = train_mobilenet_model(train_generator, validation_generator)

        # Mostrar o histórico de treinamento do MobileNet
        st.subheader("Gráfico de Acurácia e Perda do MobileNet")
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        axes[0].plot(mobilenet_history.history['accuracy'], label='Acurácia de Treinamento')
        axes[0].plot(mobilenet_history.history['val_accuracy'], label='Acurácia de Validação')
        axes[0].legend(loc='lower right')
        axes[0].set_title('Acurácia')

        axes[1].plot(mobilenet_history.history['loss'], label='Perda de Treinamento')
        axes[1].plot(mobilenet_history.history['val_loss'], label='Perda de Validação')
        axes[1].legend(loc='upper right')
        axes[1].set_title('Perda')

        st.pyplot(fig)

        # Avaliar o modelo MobileNet
        st.subheader("Avaliação do Modelo MobileNet")
        evaluate_model(mobilenet_model, validation_generator)

if __name__ == "__main__":
    main()







