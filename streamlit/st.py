import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
# Use o componente HTML para incorporar o código mermaid.js
import streamlit.components.v1 as components  # Import Streamlit
# Função para extrair o valor numérico de uma string
def extrair_valor_numerico(s):
    return int(''.join(filter(str.isdigit, s)))
st.set_page_config(layout='wide')



paginas = st.sidebar.radio(
        "Paginas👇",
        ['TEORIA',"IMAGEM", "TEXTO"],
        horizontal=False,
    )
if paginas == 'TEORIA':
    st.title('Redes GANs - TEORIA')
    
    selecao_teoria = st.radio(
            "👇",
            ["MODELO", "TREINANDO O GERADOR", "TREINANDO O DISCRIMINADOR"],
            horizontal=True,
        )
    col1,col2 = st.columns(2)
    if selecao_teoria == "MODELO":
        components.html("""         <script src="
            https://cdn.jsdelivr.net/npm/mermaid@10.6.1/dist/mermaid.min.js
            "></script>
            <pre class="mermaid">
            graph TD
            entrada_de_ruido  --> GERADOR(GERADOR)
                GERADOR  --> FAKE_IMG(FAKE_IMG)
                FAKE_IMG(FAKE_IMG) --> DISCRIMINADOR(DISCRIMINADOR)
                db[(Database)] --> REAL_IMG(REAL_IMG)
                REAL_IMG(REAL_IMG) --> DISCRIMINADOR(DISCRIMINADOR)
                DISCRIMINADOR(DISCRIMINADOR) --> CLASSIFICACAO(CLASSIFICACAO)
                CLASSIFICACAO(CLASSIFICACAO) --> LOSS(LOSS)
                LOSS(LOSS) --> GERADOR(GERADOR)
                LOSS(LOSS) --> DISCRIMINADOR(DISCRIMINADOR)
            </pre>""", width=600, height=600)
    if selecao_teoria == "TREINANDO O GERADOR":
        components.html("""         <script src="
            https://cdn.jsdelivr.net/npm/mermaid@10.6.1/dist/mermaid.min.js
            "></script>
            <pre class="mermaid">
            graph TD
                entrada_de_ruido_TRAIN(entrada_de_ruido) --> TRAIN_GERADOR(TRAIN_GERADOR)
                FIXED_WEIGHTS_DISCRIMINADOR(FIXED_WEIGHTS_DISCRIMINADOR) --> TRAIN_GERADOR(TRAIN_GERADOR)
                target_1(target_1) --> TRAIN_GERADOR(TRAIN_GERADOR)
                TRAIN_GERADOR(TRAIN_GERADOR) --o CLASSIFICACAO_TRAIN
                CLASSIFICACAO_TRAIN --> LOSS
                
            </pre>""", width=600, height=800)
    if selecao_teoria == "TREINANDO O DISCRIMINADOR":
        components.html("""         <script src="
            https://cdn.jsdelivr.net/npm/mermaid@10.6.1/dist/mermaid.min.js
            "></script>
            <pre class="mermaid">
            graph TD
                entrada_de_ruido_TRAIN(entrada_de_ruido) -->  DISCRIMINADOR(DISCRIMINADOR)
                REAL_IMG(REAL_IMG) --> DISCRIMINADOR(DISCRIMINADOR)
                FIXED_WEIGHTS_GERADOR(FIXED_WEIGHTS_GERADOR) --> DISCRIMINADOR(DISCRIMINADOR)
                target_0(target_0) --> entrada_de_ruido_TRAIN(entrada_de_ruido)
                target_1(target_1) --> REAL_IMG(REAL_IMG)
                DISCRIMINADOR(DISCRIMINADOR) --o CLASSIFICACAO_TRAIN
                CLASSIFICACAO_TRAIN --> LOSS
                
            </pre>""", width=600, height=800)


if paginas == 'IMAGEM':
    st.title('Redes GANs - IMAGEM')
    selecoes = st.radio(
            "Set label visibility 👇",
            ["GERADOR", "DISCRIMINADOR", "MODELO_COMPLETO",'TREINAMENTO'],
            horizontal=True,
        )
    if selecoes == 'TREINAMENTO':
        st.code('''
epochs = 10000
batch_size = 64
half_batch = batch_size // 2
save1 = ''
for epoch in range(epochs):
    # Treinar o discriminador
    idx = np.random.randint(0, x_train.shape[0], half_batch)
    imgs = x_train[idx]
    noise = np.random.normal(0, 1, (half_batch, latent_dim))
    gen_imgs = (generator.predict(noise)*255).astype(int)

    d_loss_real = discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
    d_loss_fake = discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # Treinar o gerador
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    valid_y = np.array([1] * batch_size)
    g_loss = gan.train_on_batch(noise, valid_y)
    if epoch %100 ==0:
        save1 += ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]\\n" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

                            ''' )
    if selecoes == 'GERADOR':
        st.code('''
        # Definir o gerador
        def build_generator(latent_dim):
            model = tf.keras.Sequential()
            model.add(layers.Dense(128, input_dim=latent_dim, activation='relu'))
            model.add(layers.Dense(784, activation='sigmoid'))
            model.add(layers.Reshape((28, 28, 1)))
            return model
                ''' )
        selecao_view = st.radio(
            "👇",
            ["MODELO", "ENTRADA", "SAÍDA"],
            horizontal=True,
        )
        if selecao_view == "MODELO":
            st.code('''
                Model: "sequential_4"
                _________________________________________________________________
                Layer (type)                Output Shape              Param #   
                =================================================================
                dense_6 (Dense)             (None, 128)               12928     
                                                                                
                dense_7 (Dense)             (None, 784)               101136    
                                                                                
                reshape_2 (Reshape)         (None, 28, 28, 1)         0         
                                                                                
                =================================================================
                Total params: 114,064
                Trainable params: 114,064
                Non-trainable params: 0
                    ''' )
        elif selecao_view == 'ENTRADA':
            col1,col2 = st.columns(2)
            with col1:
                # Generate random data for the first plot
                chart_data = pd.DataFrame(np.random.normal(0, 0.5, (100, 2)), columns=["latent_dim_count", "latent_dim_count2"])

                # Plot the first histogram
                plot_type = st.selectbox("Select Plot Type", ["Histogram", "Bar"])

                # Plot the first histogram or bar chart based on the selected option
                if plot_type == "Histogram":
                    fig = px.histogram(
                        chart_data, x=["latent_dim_count"], title='Histograma da entrada de dados'
                    )
                else:
                    fig = px.bar(
                        chart_data, y=["latent_dim_count"], title='Bar da entrada de dados'
                    )

                # Display the first plot in the first column
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                # Carregar o modelo de volta (opcional)
                loaded_model = tf.keras.models.load_model('streamlit/part1/gerador/dense1_sem_treinamento_gerador.h5')

                # Fazer previsões com o modelo carregado (apenas para teste)
                data = np.random.normal(0, 0.5, (10, 100))
                predictions = loaded_model.predict(data)
                name_columns = [f"dense{i}" for i in range(len(predictions))]
                chart_data = pd.DataFrame(predictions.T, columns=name_columns)
                # Plot the first histogram
                plot_type2 = st.selectbox("Select", ["Histogram", "Bar"])

                # Plot the first histogram or bar chart based on the selected option
                if plot_type2 == "Histogram":
                    fig2 = px.histogram(
                        chart_data, x=name_columns[1], title='Dense1'#, color_discrete_map='#fff'
                    )
                else:
                    fig2 = px.bar(
                        chart_data, y=name_columns[1], title='Dense1'#, color_discrete_map='#fff'
                    )

                # Display the second plot in the second column
                st.plotly_chart(fig2, use_container_width=True)
            loaded_model_dense2 = tf.keras.models.load_model('streamlit/part1/gerador/dense2_sem_treinamento_gerador.h5')
            # Fazer previsões com o modelo carregado (apenas para teste)
            #data = np.random.normal(0, 0.5, (10, 100))
            predictions_dense2 = loaded_model_dense2.predict(data)
            name_columns = [f"dense{i}" for i in range(len(predictions_dense2))]
            chart_data = pd.DataFrame(predictions_dense2.T, columns=name_columns)
            fig3 = px.bar(
                        chart_data, y=name_columns[1], title='Dense2'#, color_discrete_map='#fff'
            )
            st.plotly_chart(fig3, use_container_width=True)
        elif selecao_view == 'SAÍDA':
            import os
            diretorio = 'streamlit/part1/gerador/models'
            extensao_desejada = '.h5'
            # Listar apenas arquivos com a extensão desejada
            arquivos_com_extensao = [arquivo for arquivo in os.listdir(diretorio) if arquivo.endswith(extensao_desejada)]

            plot_type2 = st.selectbox("modelo", sorted(arquivos_com_extensao, key=extrair_valor_numerico))#['dense3_reshape_sem_treinamento_gerador.h5',"generator_0_MNIST.h5",
                                                #  "generator_5000_MNIST.h5","generator_20000_MNIST.h5","generator_24000_MNIST.h5",
                                                #  "basico_50k_epoca","basico_100k_epoca",
                                                #"ajustado_1k_epoca","ajustado_10k_epoca","ajustado_50k_epoca","ajustado_100k_epoca",])
            loaded_model_saida = tf.keras.models.load_model('streamlit/part1/gerador/models/'+plot_type2)
            data = np.random.normal(0, 0.5, (10, 100))
            if st.button('reload'):
                data = np.random.normal(0, 0.5, (10, 100))
            predictions = loaded_model_saida.predict(data)
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            # Criar um objeto de subplot com 1 linha e 4 colunas
            fig = make_subplots(rows=2, cols=4)
            # Adicionar cada imagem como um subplot
            cont = 0
            for i in range(2):
                for j in range(4):
                    fig.add_trace(go.Heatmap(z=(predictions[cont] * 255).reshape(28, 28), colorscale='gray', hoverinfo='text'), row=i+1, col=j + 1)
                    cont +=1
                    
            # Atualizar o layout para melhor apresentação
            fig.update_layout(width=800, height=400, showlegend=False)


            # Atualizar o layout
            fig.update_layout(
                title='Saída com 28x28',
                xaxis=dict(title='Eixo X'),
                yaxis=dict(title='amostras')
            )
            st.plotly_chart(fig, use_container_width=True)#heatmap.show()

    elif selecoes == 'DISCRIMINADOR':
        st.code('''
        # Definir o discriminador
        def build_discriminator(img_shape):
            model = tf.keras.Sequential()
            model.add(layers.Flatten(input_shape=img_shape))
            model.add(layers.Dense(128, activation='relu'))
            model.add(layers.Dense(1, activation='sigmoid'))
            return model
                ''' )
    elif selecoes == 'MODELO_COMPLETO':
        st.code('''
            # Definir a GAN combinando o gerador e o discriminador
            def build_gan(generator, discriminator):
                discriminator.trainable = False
                model = tf.keras.Sequential()
                model.add(generator)
                model.add(discriminator)
                return model
                ''' )
        
        st.code('''
               
# Configurações
latent_dim = 100
img_shape = (28, 28, 1)

# Construir e compilar o discriminador
discriminator = build_discriminator(img_shape)
discriminator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002), metrics=['accuracy'])

# Construir e compilar o gerador
generator = build_generator(latent_dim)
generator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002))

# Construir e compilar a GAN
discriminator.trainable = False
gan = build_gan(generator, discriminator)
gan.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002))
                    ''' )

if paginas == 'TEXTO':
    st.title('Gans Aplicacao de texto (Descriminalizacao do Aborto)')
    select_passos = st.radio(
        "passos👇",
        ['DATASET',"GERADOR", "DISCRIMINADOR",'APP'],
        horizontal=True,
    )
    if select_passos == 'DATASET':
        if st.button('reload (a favor)'):
            url = 'https://raw.githubusercontent.com/natorjunior/debates-ideologicos/main/dataset.csv'
            df = pd.read_csv(url)
            df = df.query('Label == 1')
            st.table(df.iloc[[np.random.randint(len(df)-1),np.random.randint(len(df)-1)]])
        if st.button('reload (contra)'):
            url = 'https://raw.githubusercontent.com/natorjunior/debates-ideologicos/main/dataset.csv'
            df = pd.read_csv(url)
            df = df.query('Label == 0')
            st.table(df.iloc[[np.random.randint(len(df)-1),np.random.randint(len(df)-1)]])
    if select_passos == 'GERADOR':
        st.code('''
            # Definir o gerador melhorado
            def build_generator(latent_dim):
                model = tf.keras.Sequential()

                model.add(layers.Dense(512, input_dim=latent_dim))
                model.add(layers.LeakyReLU(alpha=0.2))
                model.add(layers.BatchNormalization(momentum=0.8))

                model.add(layers.Dense(1024))
                model.add(layers.LeakyReLU(alpha=0.2))
                model.add(layers.BatchNormalization(momentum=0.8))

                model.add(layers.Dense(2048))
                model.add(layers.LeakyReLU(alpha=0.2))
                model.add(layers.BatchNormalization(momentum=0.8))

                model.add(layers.Dense(295, activation='softmax'))  # Ativação softmax para geração de texto
                model.add(layers.Reshape((295, 1,)))
        ''')
    if select_passos == 'DISCRIMINADOR':
            st.code('''
                    # Definir o discriminador
                def build_discriminator(img_shape):
                    model = tf.keras.Sequential()
                    model.add(layers.Flatten(input_shape=img_shape))
                    model.add(layers.Dense(128))
                    model.add(layers.LeakyReLU(alpha=0.2))
                    model.add(layers.Dense(1, activation='sigmoid'))
                    return model
            ''')
    if select_passos == 'APP':
        url = 'https://raw.githubusercontent.com/natorjunior/debates-ideologicos/main/dataset.csv'
        df = pd.read_csv(url)
        df = df.query('Label == 1')
        # Tokenizar os textos
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(df['Text'])
        vocab_size = len(tokenizer.word_index) + 1
        # Sequências de palavras para cada texto
        sequences = tokenizer.texts_to_sequences(df['Text'])

        # Padding para garantir que todas as sequências tenham o mesmo comprimento
        max_length = max(len(seq) for seq in sequences)
        padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
        # Carregar o modelo de volta (opcional)
        import os
        diretorio = 'streamlit/mnist/models_TEXT'
        extensao_desejada = '.h5'
        # Listar apenas arquivos com a extensão desejada
        arquivos_com_extensao = [arquivo for arquivo in os.listdir(diretorio) if arquivo.endswith(extensao_desejada)]

        plot_type2 = st.selectbox("modelo", sorted(arquivos_com_extensao, key=extrair_valor_numerico))#['dense3_reshape_sem_treinamento_gerador.h5',"generator_0_MNIST.h5",
                                            #  "generator_5000_MNIST.h5","generator_20000_MNIST.h5","generator_24000_MNIST.h5",
                                            #  "basico_50k_epoca","basico_100k_epoca",
                                            #"ajustado_1k_epoca","ajustado_10k_epoca","ajustado_50k_epoca","ajustado_100k_epoca",])
        loaded_model = tf.keras.models.load_model('streamlit/mnist/models_TEXT/'+plot_type2)
        #loaded_model = tf.keras.models.load_model('mnist/models_TEXT/generator_5000_TEXT.h5')
        if st.button('gerar_texto'):
            # Fazer previsões com o modelo carregado (apenas para teste)
            data = np.random.normal(0, 1, (2, 312))
            predictions = loaded_model.predict(data)
            palavra = (predictions[0]*4688).reshape(295).astype(int)
            save2 = (' '.join([tokenizer.index_word[i]  for i in palavra if i >0]))
            st.write(save2)

