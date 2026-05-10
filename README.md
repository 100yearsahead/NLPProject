# NLP Coursework Project: COGS Semantic Parsing

This is my NLP coursework project for comparing an LSTM encoder-decoder and a Transformer encoder-decoder on the COGS semantic parsing dataset.

The task is:

```text
source sentence -> target logical form

Example:

The box was found by Hannah .
-> find ( theme = * box , agent = Hannah )

The main question I looked at was whether the Transformer performs better than the LSTM on this structured semantic parsing task.

Dataset

I used the COGS dataset from Hugging Face:

GWHed/cogs

Link:

https://huggingface.co/datasets/GWHed/cogs

The dataset has three splits:

Split	Examples
train	24,155
dev	3,000
test	3,000

The scripts download the dataset automatically using the datasets library.

Models

I trained two models:

LSTM encoder-decoder
recurrent baseline
reads the sentence step by step
generates the logical form token by token
Transformer encoder-decoder
uses self-attention
can model relationships across the sentence more directly

Both models use the same preprocessing pipeline.

Final results

Final test set results:

Model	Exact Match	Token Accuracy
LSTM	83.97%	98.53%
Transformer	96.00%	99.30%

Structure-specific exact match:

Model	Passive	Clausal	Modifier
LSTM	84.73%	67.12%	57.32%
Transformer	97.17%	88.89%	85.82%

The Transformer performed better overall and on all three structure types. Modifier attachment was the hardest category for both models.

Project structure
NLPProject/
├── src/
│   ├── data.py
│   ├── vocab.py
│   ├── models.py
│   ├── train.py
│   ├── train_transformer.py
│   ├── evaluate.py
│   ├── utils.py
│   ├── inspect_predictions.py
│   └── inspect_transformer_predictions.py
├── outputs/
│   ├── checkpoints/
│   ├── tables/
│   └── figures/
├── notebooks/
│   └── notes.md
├── requirements.txt
└── README.md
Setup

I used Python 3.10.

Create a conda environment:

conda create -n nlp-cogs python=3.10 -y
conda activate nlp-cogs

Install packages:

pip install -r requirements.txt
Run final evaluation

Run these commands from the project root.

LSTM
python src/evaluate.py --model_type lstm --checkpoint outputs/checkpoints/lstm_FINAL.pt --output_csv outputs/tables/lstm_test_predictions.csv --emb_dim 128 --hidden_dim 128 --num_layers 1 --dropout 0.2 --max_decode_len 40

Expected output:

overall exact match: 0.8397
overall token accuracy: 0.9853

passive exact match: 0.8473
clausal complement exact match: 0.6712
modifier attachment exact match: 0.5732
Transformer
python src/evaluate.py --model_type transformer --checkpoint outputs/checkpoints/transformer_best.pt --output_csv outputs/tables/transformer_test_predictions.csv --emb_dim 128 --nhead 4 --num_encoder_layers 2 --num_decoder_layers 2 --dim_feedforward 256 --dropout 0.1 --max_decode_len 40 --max_len 100

Expected output:

overall exact match: 0.9600
overall token accuracy: 0.9930

passive exact match: 0.9717
clausal complement exact match: 0.8889
modifier attachment exact match: 0.8582

The prediction CSV files are created automatically when evaluate.py runs.

Training

The final trained models are already saved in outputs/checkpoints.

To retrain the LSTM:

python src/train.py --epochs 30 --batch_size 32 --emb_dim 128 --hidden_dim 128 --num_layers 1 --dropout 0.2 --lr 0.001 --teacher_forcing_ratio 0.5 --max_decode_len 40

To retrain the Transformer:

python src/train_transformer.py --epochs 20 --batch_size 32 --emb_dim 128 --nhead 4 --num_encoder_layers 2 --num