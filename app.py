import streamlit as st
import gdown
import tensorflow as tf
import io
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px

@st.cache_resource  # evitar que carregue o modelo toda vez que este arquivo alterar
def carrega_modelo():
    # https://drive.google.com/file/d/1GpWix8dp6FeFAs6g0etbnw_avu9Aflfp/view?usp=sharing
    url = 'https://drive.google.com/uc?id=1GpWix8dp6FeFAs6g0etbnw_avu9Aflfp'
    
    # Faz o download do arquivo do modelo TFLite a partir de uma URL (Google Drive)
    gdown.download(url, 'modelo_quantizado16bits.tflite')

    # Carrega o modelo TFLite (TensorFlow Lite) do caminho especificado
    interpreter = tf.lite.Interpreter(model_path='modelo_quantizado16bits.tflite')

    # Aloca os tensores no interpretador TFLite para que ele possa ser usado para inferência
    interpreter.allocate_tensors()
    
    return interpreter

def carrega_imagem():
    uploaded_file = st.file_uploader('Arraste e solte uma imagem aqui ou clique para selecionar uma', type=['png', 'jpg', 'jpeg'])

    # Verifica se o usuário fez o upload de uma imagem
    if uploaded_file is not None:
        # Lê os dados binários da imagem enviada
        image_data = uploaded_file.read()
        
        # Abre a imagem usando o PIL a partir dos dados binários
        image = Image.open(io.BytesIO(image_data))

        # Exibe a imagem na interface do Streamlit
        st.image(image)
        st.success('Imagem foi carregada com sucesso')

        # Converte a imagem para um array NumPy de ponto flutuante
        image = np.array(image, dtype=np.float32)

        # Normaliza os valores dos pixels para o intervalo [0, 1]
        image = image / 255.0

        # Adiciona uma dimensão extra para representar o batch (formato exigido pelo modelo - batch_size, altura, largura, canais)
        image = np.expand_dims(image, axis=0)

        # Retorna a imagem processada, pronta para ser usada pelo modelo
        return image
    
def previsao(interpreter, image):
    # Obtém os detalhes da entrada do modelo (por exemplo: shape, índice do tensor)
    input_details = interpreter.get_input_details()
    
    # Obtém os detalhes da saída do modelo
    output_details = interpreter.get_output_details()
    
    # Define a imagem (pré-processada) como entrada no modelo
    interpreter.set_tensor(input_details[0]['index'], image)
    
    # Executa a inferência (processo de predição) com o modelo TFLite
    interpreter.invoke()
    
    # Obtém a saída do modelo — geralmente, as probabilidades para cada classe
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Lista com os nomes das classes, correspondentes à ordem da saída do modelo
    classes = ['BlackMeasles', 'BlackRot', 'HealthyGrapes', 'LeafBlight']
    
    # Cria um DataFrame para exibir as classes e suas probabilidades
    df = pd.DataFrame()
    df['classes'] = classes
    df['probabilidades (%)'] = 100 * output_data[0]  # Converte para porcentagem
    
    # Cria um gráfico de barras horizontal com Plotly para exibir as probabilidades
    fig = px.bar(
        df,
        y='classes',
        x='probabilidades (%)',
        orientation='h',
        text='probabilidades (%)',
        title='Probabilidade de Classes de Doenças em Uvas'
    )
    
    # Exibe o gráfico interativo na interface do Streamlit
    st.plotly_chart(fig)

def main():
    st.set_page_config(
        page_title="Classifica Folhas de Videira"
    )

    st.write("# Classifica Folhas de Videira!")
    
    #Carrega modelo
    interpreter = carrega_modelo()
    
    #Carrega imagem
    image = carrega_imagem()
    
    #Classifica
    if image is not None:  
        previsao(interpreter,image) 

if __name__ == "__main__":
    main()
