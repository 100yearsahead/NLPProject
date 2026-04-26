# NLP Project Notes

## Project
**Task:** Semantic parsing on COGS  
**Dataset source:** Hugging Face (`GWHed/cogs`)  
**Models planned:** LSTM seq2seq, Transformer seq2seq

---

## Research question
How well do LSTM and Transformer sequence-to-sequence models generalise on the COGS semantic parsing task?

## Working hypothesis
The Transformer should perform better on the standard split because self-attention can model dependencies more flexibly. However, both models may still struggle when the structure becomes harder to generalise.

---

## What one example means

Example source:
`Liam hoped that a box was burned by a girl .`

Example target:
`hope ( agent = Liam , ccomp = burn ( theme = box , agent = girl ) )`

Plain-English interpretation:
- The sentence describes a **hoping event**.
- The **agent** of the hoping event is Liam.
- The content of that hope (`ccomp`) is a **burning event**.
- In the burning event, the **theme** is the box.
- In the burning event, the **agent** is the girl.

So the task is to convert a natural-language sentence into a structured meaning representation.

---

## Dataset splits

- `train`: 24,155 examples
- `dev`: 3,000 examples
- `test`: 3,000 examples

This means the dataset is large enough for a coursework comparison and small enough to train manageable models quickly.

---

## Split statistics

### Train
- `num_examples`: 24155
- `avg_src_len`: 7.48
- `avg_tgt_len`: 13.79
- `max_src_len`: 22
- `max_tgt_len`: 48
- `min_src_len`: 1
- `min_tgt_len`: 1

### Dev
- `num_examples`: 3000
- `avg_src_len`: 7.59
- `avg_tgt_len`: 13.97
- `max_src_len`: 21
- `max_tgt_len`: 48
- `min_src_len`: 3
- `min_tgt_len`: 6

### Test
- `num_examples`: 3000
- `avg_src_len`: 7.56
- `avg_tgt_len`: 14.01
- `max_src_len`: 19
- `max_tgt_len`: 40
- `min_src_len`: 3
- `min_tgt_len`: 6

---

## Meaning of the statistics

`num_examples`  
The number of sentence–logical form pairs in the split.

`avg_src_len`  
The average length of the input sentence, measured by whitespace-separated tokens.

`avg_tgt_len`  
The average length of the target logical form, also measured by whitespace-separated tokens.

`max_src_len`  
The longest input sentence in the split.

`max_tgt_len`  
The longest logical form in the split.

`min_src_len`  
The shortest input sentence in the split.

`min_tgt_len`  
The shortest logical form in the split.

---

## Interpretation of the statistics

The average input sentence is fairly short, at around 7.5 tokens, while the average logical form is almost twice as long, at around 14 tokens. This suggests that the main difficulty of the task is not handling very long sentences, but learning the correct structured mapping from sentence form to meaning representation.

The dataset is therefore a good fit for a semantic parsing comparison between two sequence-to-sequence models. It is large enough to support training and evaluation, but compact enough to allow multiple runs within a coursework timeframe.

---

## Dataset section draft sentences

The COGS dataset is a semantic parsing benchmark in which each example consists of a natural-language sentence paired with a corresponding logical form. The input is stored in the `source` field and the target meaning representation is stored in the `target` field.

Basic statistics show that the dataset contains relatively short input sequences, with an average source length of around 7.5 tokens, while target logical forms are longer, averaging around 14 tokens. This suggests that the challenge of the task lies less in long-document processing and more in learning the correct structured mapping between sentence form and semantic representation.

---

## Vocabulary / token representation

I use a simple vocabulary-based token representation rather than subword tokenisation.

Reason:
- COGS is a clean and controlled benchmark
- the target logical forms contain meaningful punctuation and symbolic structure
- whitespace tokenisation keeps the preprocessing simple and interpretable
- this is sufficient for a coursework comparison between an LSTM and a Transformer

Two vocabularies are built from the training split only:
- source vocabulary for input sentences
- target vocabulary for logical forms

Special tokens:
- <pad> for batching
- <bos> for sequence start
- <eos> for sequence end
- <unk> for unseen tokens

## Vocabulary / token representation

I use a simple vocabulary-based representation rather than subword tokenisation.

Reason:
- COGS is a clean and controlled benchmark
- the target logical forms contain meaningful punctuation and symbolic structure
- whitespace tokenisation keeps the preprocessing simple and interpretable
- this is sufficient for a coursework comparison between an LSTM and a Transformer

Two vocabularies are built from the training split only:
- source vocabulary for input sentences
- target vocabulary for logical forms

Special tokens:
- `<pad>` for batching
- `<bos>` for sequence start
- `<eos>` for sequence end
- `<unk>` for unseen tokens

### What the special tokens do

`<pad>`  
Used to pad shorter sequences so that all examples in a batch have the same length. This is necessary because PyTorch expects batched inputs to have a consistent shape.

Example:  
- Sentence 1: `Liam smiled .`  
- Sentence 2: `Emma said that Noah laughed .`

After padding, the shorter sequence might become:  
`Liam smiled . <pad> <pad> <pad>`

This lets both examples sit inside the same batch tensor.

`<bos>`  
Marks the beginning of a sequence. In a seq2seq setup, this gives the decoder a fixed starting point before it begins generating the target logical form.

Example target:  
`<bos> hope ( agent = Liam , ccomp = burn ( theme = box , agent = girl ) ) <eos>`

`<eos>`  
Marks the end of a sequence. This tells the model where the output should stop during decoding.

Example:  
If the model generates  
`hope ( agent = Liam ) <eos>`  
then decoding stops when `<eos>` is produced.

`<unk>`  
Used for tokens that were not seen in the training vocabulary. This gives the model a fallback token for unseen items rather than failing to encode them.

Example:  
If the word `Olivia` appears in test data but was not seen in training, it may be encoded as:  
`<unk>`

## Things to write later
- exact preprocessing choices
- vocabulary size
- batching and padding details
- training setup for LSTM
- training setup for Transformer
- results table
- error categories