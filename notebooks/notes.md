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

## Extra observations from random examples

Several examples show that the task is not just subject-verb-object mapping.

Examples:
- `The cloud was known .` → `know ( theme = * cloud )`
- `Aiden was given a cake .` → `give ( recipient = Aiden , theme = cake )`
- `Penelope ate the cake in the house .` → `eat ( agent = Penelope , theme = * cake ( nmod . in = * house ) )`

Observations:
- passive constructions appear in both train and evaluation splits
- the model must recover semantic roles even when surface syntax changes
- some outputs include modifier attachment, such as `nmod . in = * house`
- this suggests the challenge is structural mapping rather than long text processing


## Vocab test

I ran a small test to check whether the vocabulary pipeline works correctly on a real COGS example before moving on to batching and model training.

### Example used

**Source sentence**  
`A rose was helped by a dog .`

**Target logical form**  
`help ( theme = rose , agent = dog )`

### What the test checked

The test verified that the vocabulary class can:

- build a source vocabulary from the training `source` field
- build a target vocabulary from the training `target` field
- encode both source and target text into integer token IDs
- decode those IDs back into readable text

### Result

The source sentence encoded and decoded correctly.

Decoded source:  
`A rose was helped by a dog .`

The target logical form also encoded and decoded correctly.

Decoded target:  
`help ( theme = rose , agent = dog )`

This means the token-to-id mapping and id-to-token mapping are working as expected.

### Vocabulary sizes

- **Source vocabulary size:** 747
- **Target vocabulary size:** 662

### Interpretation of the vocabulary sizes

The vocabularies are relatively compact. This is consistent with COGS being a controlled benchmark rather than an open-domain corpus.

This suggests that the main challenge of the task is **not** handling a huge vocabulary. Instead, the difficulty lies in learning the mapping from sentence structure to logical-form structure.

### Interpreting the example

The sentence is in the **passive voice**:

`A rose was helped by a dog .`

Even though **rose** appears first in the sentence, it is not the one carrying out the action.  
The phrase **“by a dog”** shows that the **dog** is the one doing the helping.

The target logical form makes this explicit:

`help ( theme = rose , agent = dog )`

This means:

- `help` is the main event / predicate
- `agent = dog` means the dog performs the action
- `theme = rose` means the rose is the entity affected by the action

So the logical form is not preserving surface word order.  
Instead, it is preserving the underlying meaning of the sentence.

### Why this matters

This test helped confirm two important things:

1. The vocabulary pipeline is working correctly.
2. The task requires the model to recover **semantic roles** rather than rely on simple word order.

A model cannot solve this kind of example by assuming that the first noun is always the agent.  
This makes the task a genuine semantic parsing problem rather than simple pattern matching.

### What this means for the next step

Since encoding and decoding are working correctly, the next step is to use these vocabularies inside a PyTorch dataset wrapper and padded dataloader so that batches can be passed into the LSTM baseline.

## Things to write later
- exact preprocessing choices
- vocabulary size
- batching and padding details
- training setup for LSTM
- training setup for Transformer
- results table
- error categories