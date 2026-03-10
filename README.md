# Face Emotion

Reconhecimento de emoções faciais em tempo real com `MediaPipe FaceLandmarker`,
`OpenCV` e `scikit-learn`.

## Visão geral

O projeto usa o mesmo `FaceLandmarker` oficial do MediaPipe em todo o fluxo:

- `IMAGE` para extrair landmarks e blendshapes do dataset local
- `VIDEO` para processar vídeos com timestamp monotônico
- `LIVE_STREAM` para webcam com callback assíncrono

O pipeline atual treina 6 emoções:

- `Anger`
- `Disgust`
- `Fear`
- `Happiness`
- `Sadness`
- `Surprise`

Quando a confiança suavizada fica abaixo de `60%`, a aplicação mostra
`Indefinida` em vez de forçar uma emoção.

## Estrutura

```text
face-emotion/
├── assets/
│   └── face_landmarker.task
├── artifacts/
│   ├── emotion_model.joblib
│   └── cache/                # ignorado no Git
├── face_emotion/
│   ├── constants.py
│   ├── features.py
│   ├── inference.py
│   ├── landmarker.py
│   ├── smoothing.py
│   ├── training.py
│   └── visualization.py
├── reports/
│   ├── confidence_histogram.png
│   ├── confusion_matrix.png
│   ├── validation_metrics.json
│   └── validation_report.md
├── tests/
│   └── fixtures/face_sample.jpg
├── main.py
├── train.py
└── requirements.txt
```

## Requisitos

- Python 3.10 ou 3.11 para CI
- Python 3.14 validado localmente
- Webcam para o modo `live`
- Dataset local em `eINTERFACE_2021_Image/`

Observação:
`eINTERFACE_2021_Image/` permanece fora do Git e é usado apenas localmente.

## Instalação

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Treino e validação

O treino usa os splits locais `train/`, `val/` e `test/` do dataset:

```bash
python3 train.py --dataset-root eINTERFACE_2021_Image
```

Saídas principais:

- `artifacts/emotion_model.joblib`
- `reports/validation_metrics.json`
- `reports/validation_report.md`
- `reports/confusion_matrix.png`
- `reports/confidence_histogram.png`

## Resultado validado

Treino executado com o dataset local `eINTERFACE_2021_Image`:

- Modelo vencedor: `hist_gradient_boosting_blendshape`
- Accuracy em teste: `0.5884`
- Macro-F1 em teste: `0.5825`
- Cobertura de detecção:
- `train`: `5686 / 11475`
- `val`: `741 / 1433`
- `test`: `724 / 1438`

Os detalhes completos estão em [reports/validation_metrics.json](/Users/rsoares/Desktop/face-emotion-main/reports/validation_metrics.json) e [reports/validation_report.md](/Users/rsoares/Desktop/face-emotion-main/reports/validation_report.md).

## Inferência em tempo real

Modo webcam:

```bash
python3 main.py --mode live --show-mesh
```

Modo vídeo:

```bash
python3 main.py --mode video --input /caminho/video.mp4 --show-mesh
```

Parâmetros úteis:

- `--camera-index`: seleciona a webcam
- `--show-mesh`: desenha os 478 landmarks, contornos e bbox
- `--landmarker-model`: troca o `face_landmarker.task`
- `--emotion-model`: troca o classificador treinado

## Como a inferência funciona

1. O frame entra no `FaceLandmarker`
2. O sistema extrai 478 landmarks e blendshapes
3. O classificador gera probabilidades por emoção
4. As probabilidades são suavizadas com janela móvel de 5 frames válidos
5. A UI mostra rótulo, confiança em `%` e bbox/malha facial

## Testes

Executar:

```bash
pytest tests -v
```

Cobertura atual:

- normalização dos landmarks
- cálculo das features geométricas
- smoothing temporal
- limiar de confiança
- carregamento do `face_landmarker.task`
- retorno de `478` landmarks
- shape estável do vetor de features
- soma das probabilidades do artefato final

## Branches

- Base remota canônica: `roosoars/face-emotion`
- Branch de integração: `developer`
- Branch da feature: `feature/live-emotion-face-landmarker`
