# Glaucoma Detection in Optic Nerve

Este projeto utiliza modelos de aprendizado de máquina para detectar glaucoma em imagens de nervo óptico. Ele usa Streamlit para a interface do usuário, YOLO para detecção de discos e copos ópticos e XGBoost para a classificação final.

# Visão Geral
O sistema processa uma imagem de nervo óptico, detecta as regiões de disco e copo, calcula métricas importantes e, finalmente, determina a presença de glaucoma.

# Requisitos
-Python 3.7+

-Streamlit

-OpenCV

-joblib

-imutils

-numpy

-Pillow

-ultralytics (YOLO)

-xgboost

Você pode instalar todos os pacotes necessários com:

**pip install -r requirements.txt**


# Estrutura do Projeto

**glaucoma_yolov8.py:** Contém o código principal do Streamlit para carregar a imagem, processá-la e exibir os resultados.

**models/:** Diretório que contém os modelos treinados YOLO (disc.pt e cup.pt) e o modelo XGBoost (model_xgb.pkl).

**metricas_disco_escavacao.py:** Módulo com funções para calcular as métricas de escavação do disco óptico (CDR, RDR, NRR, etc.).

# Execute o aplicativo Streamlit:

Execute a aplicação por meio do {https://glaucomadetectyolov8nerveoptic.streamlit.app/}


Carregue uma imagem do nervo óptico através da interface Streamlit e veja os resultados.


