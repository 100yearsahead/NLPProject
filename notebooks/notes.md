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


## Batch sanity check

I extended the preprocessing pipeline so that COGS examples can now be turned into padded PyTorch batches.

### What this step does

At this stage, the pipeline can:

- load the COGS dataset
- compute basic split statistics
- build separate source and target vocabularies from the training split
- encode source sentences and target logical forms into token IDs
- add `<bos>` and `<eos>` markers
- pad variable-length sequences inside a batch using `<pad>`

### Batch sanity check result

Output:
- `src_ids shape: torch.Size([32, 14])`
- `tgt_ids shape: torch.Size([32, 27])`

This means:
- the batch size is 32
- the longest source sequence in this batch had length 14 after adding sequence boundary tokens
- the longest target sequence in this batch had length 27 after adding sequence boundary tokens

So the padding step is working as intended.

### Example from the batch

**Source text**  
`The cake broke .`

**Target text**  
`break ( theme = * cake )`

**Decoded source**  
`The cake broke .`

**Decoded target**  
`break ( theme = * cake )`

### Interpretation

This is a useful sanity check because it shows that the encoded sequences can be padded into tensors and then decoded back into readable text without losing the original structure. In other words, the preprocessing pipeline is not corrupting the sentence or the logical form.

The example also shows the nature of the task clearly. The sentence is short, but the target still makes the semantic role explicit by identifying `cake` as the `theme` of the `break` event. This again suggests that the difficulty of the task lies in structured meaning mapping rather than in long input sequences.

### Why this matters for the next step

This confirms that the preprocessing pipeline is ready to support model training. The next stage is to feed these padded source and target batches into the LSTM encoder-decoder baseline.

## Things to write later
- exact preprocessing choices
- vocabulary size
- batching and padding details
- training setup for LSTM
- training setup for Transformer
- results table
- error categories


## LSTM sanity check

I added a simple LSTM encoder-decoder baseline and ran a forward-pass sanity check on one batch from the COGS dataloader.

### What the test checked

The test verified that:
- the model can accept padded source and target batches
- the forward pass runs without crashing
- the output tensor has the expected shape
- greedy decoding also runs mechanically

### Output

- `src_ids shape: torch.Size([8, 14])`
- `tgt_ids shape: torch.Size([8, 29])`
- `model output shape: torch.Size([8, 29, 662])`
- `decoded shape: torch.Size([8, 20])`

### Interpretation

This means the encoder-decoder model is wired correctly.

The model output shape matches the expected seq2seq format:
- batch size = 8
- target length = 29
- target vocabulary size = 662

So for each example and each target position, the model is producing a score over the full target vocabulary.

### Example from the batch

**Source sentence**  
`A box was rented to Logan .`

**Gold target**  
`rent ( theme = box , recipient = Logan )`

**Greedy decoded output before training**  
`cart Hannah change Oliver molecule Avery Avery split tv floor dance dance dance snap Andrew coin cockroach Ella stutter`

### Why the decoded output is nonsense

This is expected because the model has not been trained yet. The weights are still random, so the decoder is just producing arbitrary target-side tokens.

The important result here is not the quality of the decoded sequence, but the fact that:
- the model runs end to end
- the output dimensions are correct
- greedy decoding works

This means the baseline architecture is ready for training.

## First LSTM training run

After 10 epochs, the LSTM baseline showed:
- train loss = 0.0969
- dev loss = 0.1184
- dev exact match = 0.0000
- dev token accuracy = 0.9801

Interpretation:
The model is clearly learning at the token level, as shown by the low training/dev loss and very high token accuracy. However, exact-match accuracy remains at zero, which means the model is not yet producing fully correct logical forms under greedy decoding.

This suggests a gap between token-level learning and sequence-level generation. Since exact match is very strict, even small structural mistakes such as one wrong token, an extra bracket, missing punctuation, or an early decoding error can reduce the sequence score to zero.

The next step is to inspect actual predictions on development examples to understand whether the model is producing almost-correct outputs or failing in a more systematic way.

## First inspection of LSTM dev predictions

The first inspected dev predictions show that the LSTM baseline is learning much more than the exact-match score alone suggests.

Two main patterns appear:

1. The model often predicts the overall logical-form structure correctly, including the main predicate and role labels such as `agent`, `theme`, and `recipient`.
2. Many predictions continue generating extra closing brackets `)` after the logical form is already essentially complete.

This helps explain the gap between high token accuracy and zero exact match. Since exact match is a strict sequence-level metric, a prediction can be almost fully correct but still count as wrong if it contains a small number of extra tokens at the end.

A second pattern is occasional lexical substitution inside an otherwise correct structure. For example, the model sometimes predicts the wrong predicate or replaces one entity with another while keeping the broader logical-form frame intact.

Overall, the early LSTM predictions suggest that the model is learning the target format and much of the role structure, but is still weak at exact sequence termination and fully correct lexical recovery.
================================================================================
Example 1
SOURCE:
Liam hoped that a box was burned by a girl .

GOLD:
hope ( agent = Liam , ccomp = burn ( theme = box , agent = girl ) )

PRED:
hope ( agent = Liam , ccomp = roll ( theme = box , agent = girl ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) )

================================================================================
Example 2
SOURCE:
The donkey lended the cookie to a mother .

GOLD:
lend ( agent = * donkey , theme = * cookie , recipient = mother )

PRED:
lend ( agent = * lawyer , theme = * cookie , recipient = baby ) ) ) ) ) ) ) ) )

================================================================================
Example 3
SOURCE:
A melon was given to a girl by the guard .

GOLD:
give ( theme = melon , recipient = girl , agent = * guard )

PRED:
give ( theme = melon , recipient = girl , agent = * mouse ) ) ) ) ) ) ) ) )

================================================================================
Example 4
SOURCE:
A donut was given to a butterfly .

GOLD:
give ( theme = donut , recipient = butterfly )

PRED:
give ( theme = donut , recipient = lion ) ) ) ) ) ) ) ) ) )

================================================================================
Example 5
SOURCE:
A rose was mailed to Isabella .

GOLD:
mail ( theme = rose , recipient = Isabella )

PRED:
mail ( theme = rose , recipient = Isabella ) ) ) ) ) ) ) ) ) ) ) ) )

================================================================================
Example 6
SOURCE:
The girl offered the weapon beside a machine to a chicken .

GOLD:
offer ( agent = * girl , theme = * weapon ( nmod . beside = machine ) , recipient = chicken )

PRED:
offer ( agent = * girl , theme = * game ( nmod . beside = chair ) , recipient = chicken ) ) ) ) ) ) ) ) ) ) ) )

================================================================================
Example 7
SOURCE:
A donut was touched by Emma .

GOLD:
touch ( theme = donut , agent = Emma )

PRED:
touch ( theme = donut , agent = Emma ) ) ) ) ) ) ) ) ) ) ) ) )

================================================================================
Example 8
SOURCE:
Liam painted a box on a table beside the chair .

GOLD:
paint ( agent = Liam , theme = box ( nmod . on = table ( nmod . beside = * chair ) ) )

PRED:
paint ( agent = Liam , theme = box ( nmod . on = table ( nmod . beside = * ) ) ) ) ) ) ) ) ) ) ) ) )

================================================================================
Example 9
SOURCE:
Emma ate a hammer .

GOLD:
eat ( agent = Emma , theme = hammer )

PRED:
eat ( agent = Emma , theme = molecule ) ) ) ) ) ) ) ) ) ) )

================================================================================
Example 10
SOURCE:
A drink was juggled .

GOLD:
juggle ( theme = drink )

PRED:
juggle ( theme = drink ) ) ) ) ) ) ) ) ) ) ) ) ) ) )

================================================================================
Example 11
SOURCE:
Liam liked that the cake was tossed by Ava .

GOLD:
like ( agent = Liam , ccomp = toss ( theme = * cake , agent = Ava ) )

PRED:
like ( agent = Liam , ccomp = draw ( theme = * cake , agent = Ava ) ) ) ) ) ) ) ) ) ) )

================================================================================
Example 12
SOURCE:
The box was found by Hannah .

GOLD:
find ( theme = * box , agent = Hannah )

PRED:
find ( theme = * box , agent = Lucas ) ) ) ) ) ) ) ) ) )

================================================================================
Example 13
SOURCE:
Ava forwarded a chalk to a girl .

GOLD:
forward ( agent = Ava , theme = chalk , recipient = girl )

PRED:
forward ( agent = Ava , theme = chalk , recipient = girl ) ) ) ) ) ) ) ) ) ) ) ) ) )

================================================================================
Example 14
SOURCE:
Mason liked that the cookie was hunted .

GOLD:
like ( agent = Mason , ccomp = hunt ( theme = * cookie ) )

PRED:
like ( agent = Mason , ccomp = eat ( theme = * cookie ) ) ) ) ) ) ) ) ) )

================================================================================
Example 15
SOURCE:
A monkey ran .

GOLD:
run ( agent = monkey )

PRED:
run ( agent = monkey ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) )


## Inspection of best LSTM checkpoint

I inspected 20 development-set predictions from the best saved LSTM checkpoint to understand what the model is actually getting right and wrong.

### Overall impression

The LSTM is learning the task much better than the earliest exact-match scores suggested.

From this sample:
- many predictions are fully correct
- many near-misses are caused by extra closing brackets `)` at the end
- some predictions keep the full logical-form structure but substitute the wrong predicate or entity

This suggests that the model has learned much of the output format and semantic role structure, but still has weaknesses in sequence termination and exact lexical recovery.

---

## Error type 1: exact matches

Several development examples are predicted perfectly.

### Example 1
**Source**  
`Liam hoped that a box was burned by a girl .`

**Gold**  
`hope ( agent = Liam , ccomp = burn ( theme = box , agent = girl ) )`

**Prediction**  
`hope ( agent = Liam , ccomp = burn ( theme = box , agent = girl ) )`

**Interpretation**  
This is a strong result because it is not a simple flat sentence. The model correctly identifies:
- the main event: `hope`
- the agent of the main event: `Liam`
- the embedded event through `ccomp = burn`
- the roles inside the embedded event: `theme = box`, `agent = girl`

So the model is clearly capable of generating correct nested event structure.

### Example 2
**Source**  
`A melon was given to a girl by the guard .`

**Gold**  
`give ( theme = melon , recipient = girl , agent = * guard )`

**Prediction**  
`give ( theme = melon , recipient = girl , agent = * guard )`

**Interpretation**  
This is a passive construction. The sentence begins with `melon`, but the model still correctly recovers that `guard` is the agent. This shows that the model is not relying only on surface word order.

### Example 3
**Source**  
`Emma gave the girl a cake .`

**Gold**  
`give ( agent = Emma , recipient = * girl , theme = cake )`

**Prediction**  
`give ( agent = Emma , recipient = * girl , theme = cake )`

**Interpretation**  
This is a clean example of correct argument-role assignment. The model correctly separates:
- `Emma` as the agent
- `girl` as the recipient
- `cake` as the theme

---

## Error type 2: over-generation / stopping failure

A common error is that the model produces an almost completely correct logical form, but then continues generating extra closing brackets `)`.

### Example 4
**Source**  
`A rose was mailed to Isabella .`

**Gold**  
`mail ( theme = rose , recipient = Isabella )`

**Prediction**  
`mail ( theme = rose , recipient = Isabella ) ) ) ) ) ) ) ) ) ) ) ) )`

**Interpretation**  
The core logical form is correct. The error is not semantic structure, but failure to stop decoding at the right point. This is important because exact-match evaluation is strict, so a nearly perfect prediction still counts as wrong if extra tokens are added at the end.

### Example 5
**Source**  
`A monkey ran .`

**Gold**  
`run ( agent = monkey )`

**Prediction**  
`run ( agent = monkey ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) )`

**Interpretation**  
Again, the model identifies the correct predicate and role assignment, but continues generating after the output is already complete.

### Example 6
**Source**  
`A donut was given to a butterfly .`

**Gold**  
`give ( theme = donut , recipient = butterfly )`

**Prediction**  
`give ( theme = donut , recipient = butterfly ) )`

**Interpretation**  
This is another example where the structure is correct but exact match is lost because of a small decoding error at the end.

---

## Error type 3: lexical substitution inside a mostly correct structure

Another error pattern is that the model keeps the general logical-form structure but inserts the wrong predicate or entity in one slot.

### Example 7
**Source**  
`The girl offered the weapon beside a machine to a chicken .`

**Gold**  
`offer ( agent = * girl , theme = * weapon ( nmod . beside = machine ) , recipient = chicken )`

**Prediction**  
`offer ( agent = * girl , theme = * weapon ( nmod . beside = car ) , recipient = chicken )`

**Interpretation**  
The model gets the overall structure right:
- main predicate: `offer`
- agent: `girl`
- theme: `weapon`
- recipient: `chicken`
- modifier structure: `nmod . beside = ...`

The mistake is lexical: `car` is predicted instead of `machine`. This suggests that the model has learned the structural frame better than the exact lexical content.

### Example 8
**Source**  
`Liam liked that the cake was tossed by Ava .`

**Gold**  
`like ( agent = Liam , ccomp = toss ( theme = * cake , agent = Ava ) )`

**Prediction**  
`like ( agent = Liam , ccomp = hold ( theme = * cake , agent = Ava ) )`

**Interpretation**  
The model correctly predicts:
- the main event: `like`
- the embedded event through `ccomp =`
- the correct embedded argument roles

But it chooses the wrong embedded predicate: `hold` instead of `toss`. Again, the structure is mostly right but one key lexical choice is wrong.

### Example 9
**Source**  
`Liam was slipped a cloud in the room by the chicken .`

**Gold**  
`slip ( recipient = Liam , theme = cloud ( nmod . in = * room ) , agent = * chicken )`

**Prediction**  
`slip ( recipient = Liam , theme = can ( nmod . in = * room ) , agent = * chicken )`

**Interpretation**  
This is a more complex example involving passive structure and modifier attachment. The model gets:
- the main predicate: `slip`
- the recipient: `Liam`
- the location modifier structure: `nmod . in = * room`
- the agent: `chicken`

But it substitutes `can` for `cloud`. This again supports the idea that the structural representation is being learned more successfully than the exact lexical item.

---

## Main takeaway from these examples

These inspected predictions suggest that the LSTM baseline is learning the target logical-form structure reasonably well.

The main remaining weaknesses are:
1. **Stopping behaviour** — the model often generates extra `)` after an otherwise correct output.
2. **Lexical substitution** — the model sometimes predicts the wrong predicate or entity inside an otherwise correct structure.

This helps explain why token accuracy becomes very high before exact-match accuracy becomes stable. The model is often very close to the correct answer, but exact match is strict enough that small sequence-level mistakes still count as total errors.

## LSTM experiment progression and reasoning

The LSTM experiments were not chosen as a broad hyperparameter sweep. Instead, they were guided by what the model was actually doing in its first outputs. The goal was to start with a standard recurrent baseline, inspect its behaviour, identify the main weaknesses, and then run a small number of targeted follow-up experiments.

---

## First LSTM baseline: what happened

The first LSTM baseline used a simple and standard configuration:

- embedding dimension = 128
- hidden dimension = 256
- number of layers = 1
- dropout = 0.2
- learning rate = 0.001
- batch size = 32
- teacher forcing ratio = 0.5
- max decode length = 40

This was chosen as a safe starting point because:
- COGS has relatively short source sequences
- the vocabulary is compact
- the task is structured sequence generation rather than long-context modelling
- the aim at this stage was to get a stable recurrent baseline rather than optimise aggressively

The first full training run showed that:
- train loss decreased steadily
- dev loss decreased steadily
- token accuracy became very high
- exact match was much harder and more unstable

At first, this looked confusing because the model seemed to be learning well at the token level but still getting very poor exact-match results. To understand this, I inspected actual development-set predictions.

---

## What inspection of the first outputs showed

The first prediction inspection was very revealing. It showed that the LSTM was often producing outputs that were *almost correct*, but then continuing generation after the logical form should already have stopped.

### Example: passive sentence with over-generation

**Source**  
`A rose was mailed to Isabella .`

**Gold**  
`mail ( theme = rose , recipient = Isabella )`

**Prediction**  
`mail ( theme = rose , recipient = Isabella ) ) ) ) ) ) ) ) ) ) ) ) )`

### Interpretation

This prediction is almost correct. The model:
- identified the predicate `mail`
- assigned the correct `theme`
- assigned the correct `recipient`

The real problem is that it did not stop generation cleanly. It continued producing extra closing brackets `)` after the logical form was already complete.

This mattered because exact match is a strict metric. Even if the whole logical form is right apart from a few extra symbols at the end, the prediction still counts as fully wrong.

---

### Another example: simple intransitive sentence with the same problem

**Source**  
`A monkey ran .`

**Gold**  
`run ( agent = monkey )`

**Prediction**  
`run ( agent = monkey ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) )`

### Interpretation

Again, the model clearly understood the sentence:
- correct predicate `run`
- correct agent `monkey`

But it kept decoding too long.

This suggested that the model had already learned much of the logical-form structure, but was still weak at sequence termination.

---

### Example: nested event structure mostly learned

**Source**  
`Liam hoped that a box was burned by a girl .`

**Gold**  
`hope ( agent = Liam , ccomp = burn ( theme = box , agent = girl ) )`

**Prediction from later strong checkpoint**  
`hope ( agent = Liam , ccomp = burn ( theme = box , agent = girl ) )`

### Interpretation

This kind of example showed that the model was not just learning flat templates. It could recover:
- the main event `hope`
- the embedded event `burn`
- the correct role structure inside the embedded event

That was an important sign that the LSTM baseline was learning real structure, even if exact match was initially hiding this.

---

## Main lesson from the first inspection

The first inspection suggested three important things:

### 1. The model was learning more than exact match alone suggested
The LSTM was often producing the correct frame:
- correct predicate
- correct role labels
- mostly correct logical-form structure

### 2. Sequence termination was a major issue
The model often failed by generating extra `)` at the end.

### 3. Some lexical substitutions remained
In some examples, the broad structure was correct but a predicate or entity was wrong.

For example:

**Source**  
`Liam liked that the cake was tossed by Ava .`

**Gold**  
`like ( agent = Liam , ccomp = toss ( theme = * cake , agent = Ava ) )`

**Prediction from earlier run**  
`like ( agent = Liam , ccomp = hold ( theme = * cake , agent = Ava ) )`

Here the model:
- got the outer event right
- got the embedded structure right
- got the argument roles right

but chose the wrong embedded predicate.

This suggested that the model was learning the structural frame faster than exact lexical recovery.

---

## Why the later LSTM experiments were chosen

The later experiments were chosen to respond directly to what the first inspected outputs showed.

The point was not to try lots of random hyperparameters. The point was to test a few specific hypotheses.

---

## Experiment 1: baseline

### Why it was run
This was the standard first recurrent baseline.

### What question it answered
Can a simple LSTM encoder-decoder learn the COGS semantic parsing task at all?

### What it showed
Yes. It clearly learned:
- much of the output format
- role labels such as `agent`, `theme`, and `recipient`
- some nested and passive structures

But it also showed:
- over-generation
- unstable exact match
- some lexical substitution

This inspection shaped the rest of the experiments.

---

## Experiment 2: increased teacher forcing

### What changed
- teacher forcing ratio increased from 0.5 to 0.7

### Why this was a sensible next experiment
Teacher forcing controls how often the decoder is trained using the true previous target token instead of its own previous prediction.

Since the first baseline often produced:
- almost correct logical forms
- then failed at the end with extra `)`
- or drifted after a mostly correct prefix

it was reasonable to test whether giving the decoder more guidance during training would improve sequence-level generation.

### What question it was testing
Would stronger decoder guidance reduce over-generation and produce cleaner full logical forms?

### Why this follows from the first inspection
Because the main visible weakness in the early predictions was not complete structural collapse. It was that the decoder often continued after the right answer was already produced. Teacher forcing is directly relevant to this kind of problem.

---

## Experiment 3: reduced hidden dimension

### What changed
- hidden dimension reduced from 256 to 128

### Why this was a sensible next experiment
The dataset is:
- controlled
- relatively small
- short in sequence length
- compact in vocabulary

So it was reasonable to ask whether the larger recurrent state was actually helping, or whether it was making the model less clean and less stable.

### What question it was testing
Does the task really need the larger recurrent memory, or does a smaller LSTM generalise more cleanly?

### Why this follows from the first inspection
The first baseline showed that the model could already produce much of the structure correctly. That suggested the problem might not be a lack of capacity. It might instead be that a smaller model would be enough and might even produce cleaner generation.

This turned out to be an important experiment, because the smaller hidden-size model gave the strongest LSTM results.

---

## Experiment 4: dropout sensitivity

### What changed
- dropout varied around the best general region

### Why this was tested later, not earlier
Dropout is mainly about regularisation. It matters if:
- the model is memorising too much
- the model is brittle
- training and dev behaviour diverge in an overfitting-like way

But the first inspected outputs suggested that the more urgent issues were:
- stopping behaviour
- sequence generation quality
- exact lexical recovery

Those are not obviously the first places where dropout should be blamed. That is why dropout was tested later, after teacher forcing and hidden size.

### What question it was testing
Is the LSTM strongly sensitive to regularisation, or are the more important factors decoder guidance and model capacity?

### What happened
The dropout changes did not produce improvements comparable to the hidden-size reduction. That suggests dropout was not the main driver of performance in this setup.

---

## Why the order of experiments made sense

The order of experiments was important.

### Step 1: establish a real baseline
Start with a standard LSTM setup.

### Step 2: inspect actual predictions
Do not trust exact-match numbers alone. Look at what the model is really outputting.

### Step 3: choose targeted follow-up runs
Use the observed error patterns to decide what to test:
- more teacher forcing because stopping/generation looked weak
- smaller hidden state because the task might not need as much capacity
- dropout later, because regularisation did not look like the first bottleneck

### Step 4: extend the strongest configuration
Once the hidden-dimension-128 model looked strongest and cleaner, extend it to 30 epochs to check whether exact-match performance was still improving.

This progression is much more defensible than running many random hyperparameter combinations.

---

## What the strongest LSTM run showed

The strongest LSTM run used:
- embedding dimension = 128
- hidden dimension = 128
- number of layers = 1
- dropout = 0.2
- learning rate = 0.001
- batch size = 32
- teacher forcing ratio = 0.5
- max decode length = 40

This configuration produced much cleaner predictions.

### Example: exact passive prediction

**Source**  
`A melon was given to a girl by the guard .`

**Gold**  
`give ( theme = melon , recipient = girl , agent = * guard )`

**Prediction**  
`give ( theme = melon , recipient = girl , agent = * guard )`

### Interpretation

This is a strong example because it is not just a simple active sentence. It involves:
- passive structure
- correct role recovery
- correct ordering inside the logical form

The model gets the whole structure right.

---

### Example: exact embedded event prediction

**Source**  
`Emma preferred to giggle .`

**Gold**  
`prefer ( agent = Emma , xcomp = giggle ( agent = Emma ) )`

**Prediction**  
`prefer ( agent = Emma , xcomp = giggle ( agent = Emma ) )`

### Interpretation

This shows that the LSTM can also handle embedded event structure, not just flat predicate-argument forms.

---

### Example: remaining lexical substitution error

**Source**  
`The girl offered the weapon beside a machine to a chicken .`

**Gold**  
`offer ( agent = * girl , theme = * weapon ( nmod . beside = machine ) , recipient = chicken )`

**Prediction**  
`offer ( agent = * girl , theme = * shell ( nmod . beside = car ) , recipient = dog )`

### Interpretation

This prediction shows the remaining weakness clearly:
- the broader frame is still recognisably correct
- the model knows this is an `offer` event
- it uses the right role labels
- it preserves the modifier structure

But several lexical items are wrong:
- `shell` instead of `weapon`
- `car` instead of `machine`
- `dog` instead of `chicken`

This supports the idea that, by this point, the model is learning structure more reliably than exact lexical content.

---

## Main takeaway from the LSTM study

The LSTM experiments suggest that:
- the recurrent baseline can learn a substantial amount of the semantic structure in COGS
- exact-match evaluation can underestimate what the model has learned when generation errors are small but strict
- reducing model capacity from hidden size 256 to 128 improved the recurrent baseline substantially
- teacher forcing also mattered, but less than the capacity change
- dropout was not the main factor driving performance in this setup

The most important lesson is that the model’s behaviour had to be interpreted through both metrics and inspected predictions. Looking only at exact match would have hidden the fact that the model was often producing nearly correct or fully correct logical forms.

One issue identified in the initial LSTM implementation was that batched greedy decoding only stopped when all sequences in the batch predicted `<eos>`. This meant that sequences which had already finished could continue generating extra tokens, which likely contributed to over-generation errors such as repeated closing brackets. The decoding function was therefore updated to track finished sequences individually.

The important catch

Your draft currently says:

“Bahdanau, Cho and Bengio (2015) showed that attention improves encoder-decoder models...”

That is fine in the related work section.

But your actual implemented LSTM is not Bahdanau attention.
It is a plain seq2seq LSTM.

So you should be careful not to imply that your baseline uses attention if it does not.


## Final corrected LSTM run

After fixing the batched greedy-decoding issue, I reran the strongest LSTM configuration.

Best result:
- best epoch = 29
- train loss = 0.0378
- dev loss = 0.1080
- dev exact match = 0.8440
- dev token accuracy = 0.9866

Interpretation:
The corrected run produced a smooth and believable training curve. Development exact-match accuracy increased steadily across epochs, which suggests that the earlier unstable sequence-level behaviour was largely caused by the decoding bug rather than by the model itself. This corrected LSTM run is therefore a much more reliable baseline.

Conclusion:
The hidden-dimension-128 LSTM checkpoint from epoch 29 will be kept as the final recurrent baseline for comparison against the Transformer model.


================================================================================
Example 1
SOURCE:
Liam hoped that a box was burned by a girl .

GOLD:
hope ( agent = Liam , ccomp = burn ( theme = box , agent = girl ) )

PRED:
hope ( agent = Liam , ccomp = burn ( theme = box , agent = girl ) )

================================================================================
Example 2
SOURCE:
The donkey lended the cookie to a mother .

GOLD:
lend ( agent = * donkey , theme = * cookie , recipient = mother )

PRED:
lend ( agent = * donkey , theme = * key , recipient = driver )

================================================================================
Example 3
SOURCE:
A melon was given to a girl by the guard .

GOLD:
give ( theme = melon , recipient = girl , agent = * guard )

PRED:
give ( theme = melon , recipient = girl , agent = * driver )

================================================================================
Example 4
SOURCE:
A donut was given to a butterfly .

GOLD:
give ( theme = donut , recipient = butterfly )

PRED:
give ( theme = donut , recipient = butterfly )

================================================================================
Example 5
SOURCE:
A rose was mailed to Isabella .

GOLD:
mail ( theme = rose , recipient = Isabella )

PRED:
mail ( theme = rose , recipient = Isabella )

================================================================================
Example 6
SOURCE:
The girl offered the weapon beside a machine to a chicken .

GOLD:
offer ( agent = * girl , theme = * weapon ( nmod . beside = machine ) , recipient = chicken )

PRED:
offer ( agent = * girl , theme = * weapon ( nmod . beside = bed ) , recipient = chicken )

================================================================================
Example 7
SOURCE:
A donut was touched by Emma .

GOLD:
touch ( theme = donut , agent = Emma )

PRED:
touch ( theme = donut , agent = Emma )

================================================================================
Example 8
SOURCE:
Liam painted a box on a table beside the chair .

GOLD:
paint ( agent = Liam , theme = box ( nmod . on = table ( nmod . beside = * chair ) ) )

PRED:
paint ( agent = Liam , theme = box ( nmod . on = table ( nmod . beside = * chair ) ) )

================================================================================
Example 9
SOURCE:
Emma ate a hammer .

GOLD:
eat ( agent = Emma , theme = hammer )

PRED:
eat ( agent = Emma , theme = hammer )

================================================================================
Example 10
SOURCE:
A drink was juggled .

GOLD:
juggle ( theme = drink )

PRED:
juggle ( theme = drink )

================================================================================
Example 11
SOURCE:
Liam liked that the cake was tossed by Ava .

GOLD:
like ( agent = Liam , ccomp = toss ( theme = * cake , agent = Ava ) )

PRED:
like ( agent = Liam , ccomp = toss ( theme = * cake , agent = Ava ) )

================================================================================
Example 12
SOURCE:
The box was found by Hannah .

GOLD:
find ( theme = * box , agent = Hannah )

PRED:
find ( theme = * box , agent = Hannah )

================================================================================
Example 13
SOURCE:
Ava forwarded a chalk to a girl .

GOLD:
forward ( agent = Ava , theme = chalk , recipient = girl )

PRED:
forward ( agent = Ava , theme = chalk , recipient = girl )

================================================================================
Example 14
SOURCE:
Mason liked that the cookie was hunted .

GOLD:
like ( agent = Mason , ccomp = hunt ( theme = * cookie ) )

PRED:
like ( agent = Mason , ccomp = hunt ( theme = * cookie ) )

================================================================================
Example 15
SOURCE:
A monkey ran .

GOLD:
run ( agent = monkey )

PRED:
run ( agent = monkey )

## Final LSTM checkpoint inspection

I inspected 15 development examples from the corrected best LSTM checkpoint.

### Overall impression

The predictions are much cleaner than in the earlier inspected run. In particular:
- the repeated extra `)` problem is no longer visible
- many outputs are now fully exact
- the main remaining errors are lexical substitutions rather than structural collapse

From this sample, 11 out of 15 predictions were exact.

### Examples of exact predictions

**Source**  
`Liam hoped that a box was burned by a girl .`

**Gold**  
`hope ( agent = Liam , ccomp = burn ( theme = box , agent = girl ) )`

**Prediction**  
`hope ( agent = Liam , ccomp = burn ( theme = box , agent = girl ) )`

This shows the model can recover nested event structure correctly.

---

**Source**  
`Liam painted a box on a table beside the chair .`

**Gold**  
`paint ( agent = Liam , theme = box ( nmod . on = table ( nmod . beside = * chair ) ) )`

**Prediction**  
`paint ( agent = Liam , theme = box ( nmod . on = table ( nmod . beside = * chair ) ) )`

This shows the model can also handle modifier attachment correctly.

---

**Source**  
`Liam liked that the cake was tossed by Ava .`

**Gold**  
`like ( agent = Liam , ccomp = toss ( theme = * cake , agent = Ava ) )`

**Prediction**  
`like ( agent = Liam , ccomp = toss ( theme = * cake , agent = Ava ) )`

This is another strong clausal-complement example.

### Examples of remaining errors

**Source**  
`The donkey lended the cookie to a mother .`

**Gold**  
`lend ( agent = * donkey , theme = * cookie , recipient = mother )`

**Prediction**  
`lend ( agent = * donkey , theme = * key , recipient = driver )`

The overall structure is correct, but the model substitutes the wrong lexical items.

---

**Source**  
`A melon was given to a girl by the guard .`

**Gold**  
`give ( theme = melon , recipient = girl , agent = * guard )`

**Prediction**  
`give ( theme = melon , recipient = girl , agent = * driver )`

Again, the predicate and role structure are correct, but the final entity is wrong.

---

**Source**  
`The girl offered the weapon beside a machine to a chicken .`

**Gold**  
`offer ( agent = * girl , theme = * weapon ( nmod . beside = machine ) , recipient = chicken )`

**Prediction**  
`offer ( agent = * girl , theme = * weapon ( nmod . beside = bed ) , recipient = chicken )`

Here the modifier structure is preserved, but one lexical item inside it is wrong.

### Main takeaway

The corrected LSTM baseline now appears to learn the target logical-form structure reliably. The dominant remaining weakness is lexical substitution inside an otherwise correct structure, rather than output-format instability or major structural failure.

## Why greedy decoding was used

Since this project is a sequence-to-sequence generation task, I needed an inference procedure at evaluation time to turn model outputs into full logical forms. During training, the model sees the gold target prefix, but at evaluation time it has to generate the target sequence using its own previous predictions. Greedy decoding provides a simple and standard way to do this.

In greedy decoding, the model starts from the `<bos>` token and generates one token at a time. At each step, it chooses the single most likely next token, appends it to the sequence, and then uses that token as part of the next decoder input. Generation stops when `<eos>` is produced or when a maximum decode length is reached.

I used greedy decoding because:
- it is a standard baseline inference method for seq2seq models
- it is simple to implement and explain
- it can be used consistently for both the LSTM and Transformer
- it allows direct evaluation of fully generated logical forms using exact match and token accuracy

This was especially important in this project because token-level metrics alone did not fully reflect model behaviour. Greedy decoding made it possible to inspect actual generated logical forms and identify sequence-level issues such as over-generation, stopping errors, and lexical substitutions.

## Transformer sanity check

I added a Transformer encoder-decoder model and ran a forward-pass sanity check on one batch from the COGS dataloader.

### Output

- `src_ids shape: torch.Size([8, 15])`
- `tgt_ids shape: torch.Size([8, 28])`
- `model output shape: torch.Size([8, 28, 662])`
- `decoded shape: torch.Size([8, 20])`

### Interpretation

This confirms that the Transformer accepts padded source and target batches, produces an output tensor in the expected seq2seq format, and supports greedy decoding at inference time.

As expected, the generated output was not meaningful at this stage because the model is still untrained. The purpose of this check was only to confirm that the Transformer is wired correctly and is ready for training.

## Transformer hyperparameters and intuition

The Transformer uses a small encoder-decoder configuration so that the comparison with the LSTM remains fair and interpretable.

Main hyperparameters:
- `emb_dim`: size of the token and hidden representations
- `nhead`: number of attention heads
- `num_encoder_layers`: number of stacked encoder blocks
- `num_decoder_layers`: number of stacked decoder blocks
- `dim_feedforward`: hidden size of the feedforward layer inside each Transformer block
- `dropout`: regularisation strength
- `lr`: learning rate
- `batch_size`: number of examples per update
- `max_decode_len`: maximum output length during greedy decoding

Initial values:
- `emb_dim=128`
- `nhead=4`
- `num_encoder_layers=2`
- `num_decoder_layers=2`
- `dim_feedforward=256`
- `dropout=0.2`
- `lr=0.0005`
- `batch_size=32`

Reasoning:
These choices keep the Transformer relatively small and comparable to the LSTM baseline, while still allowing multi-head attention and a real encoder-decoder structure. The model is large enough to capture structured dependencies but not so large that it becomes unnecessarily difficult to train on COGS.


## First Transformer training run

The first Transformer run trained much faster than the LSTM baseline in the early epochs and reached a best development exact-match score of 0.8310 at epoch 7.

Best result:
- best epoch = 7
- train loss = 0.1447
- dev loss = 0.0679
- dev exact match = 0.8310
- dev token accuracy = 0.9542

Interpretation:
The Transformer learned the task quickly and produced a strong development exact-match score within a small number of epochs. However, after the best checkpoint, exact-match performance became less stable and did not continue improving. At this stage, the Transformer is competitive with the corrected LSTM baseline but does not yet clearly outperform it.

Next step:
Inspect predictions from the best checkpoint and run one small follow-up experiment focused on training stability, most likely a reduced dropout setting.

## Inspection of best Transformer checkpoint

I inspected 15 development examples from the best Transformer checkpoint.

### Overall impression

The Transformer predictions are generally very clean. In this sample, 12 out of 15 predictions were exact. The model handles passive constructions, clausal complements, and simple predicate-argument structure well.

### Examples of exact predictions

**Source**  
`Liam hoped that a box was burned by a girl .`

**Gold**  
`hope ( agent = Liam , ccomp = burn ( theme = box , agent = girl ) )`

**Prediction**  
`hope ( agent = Liam , ccomp = burn ( theme = box , agent = girl ) )`

This shows that the Transformer can recover nested event structure correctly.

---

**Source**  
`A melon was given to a girl by the guard .`

**Gold**  
`give ( theme = melon , recipient = girl , agent = * guard )`

**Prediction**  
`give ( theme = melon , recipient = girl , agent = * guard )`

This shows correct passive role assignment.

---

**Source**  
`Liam liked that the cake was tossed by Ava .`

**Gold**  
`like ( agent = Liam , ccomp = toss ( theme = * cake , agent = Ava ) )`

**Prediction**  
`like ( agent = Liam , ccomp = toss ( theme = * cake , agent = Ava ) )`

This is another strong clausal-complement example.

### Examples of remaining errors

**Source**  
`The girl offered the weapon beside a machine to a chicken .`

**Gold**  
`offer ( agent = * girl , theme = * weapon ( nmod . beside = machine ) , recipient = chicken )`

**Prediction**  
`offer ( agent = * girl , theme = weapon ( nmod . beside = machine ) , recipient = chicken )`

This prediction is nearly correct, but misses the `*` marker before `weapon`.

---

**Source**  
`Liam painted a box on a table beside the chair .`

**Gold**  
`paint ( agent = Liam , theme = box ( nmod . on = table ( nmod . beside = * chair ) ) )`

**Prediction**  
`paint ( agent = Liam , theme = box ( nmod . beside = * table ( nmod . beside = table ) )`

This looks like a modifier attachment error, which is one of the more structurally difficult cases in COGS.

---

**Source**  
`Mason liked that the cookie was hunted .`

**Gold**  
`like ( agent = Mason , ccomp = hunt ( theme = * cookie ) )`

**Prediction**  
`like ( agent = Mason , ccomp = hunt ( theme = * cookie )`

This is almost correct, but the logical form is incomplete because of a missing closing bracket.

### Main takeaway

The Transformer is structurally strong and produces many fully correct logical forms. The remaining errors appear to be mainly small formatting issues and some attachment mistakes, rather than broad failure to recover the semantic frame.

## Final Transformer result

The strongest Transformer run achieved a best development exact-match score of 0.9677 at epoch 19.

Best result:
- best epoch = 19
- train loss = 0.0266
- dev loss = 0.0145
- dev exact match = 0.9677
- dev token accuracy = 0.9935

Interpretation:
This run clearly outperformed the final corrected LSTM baseline on the development set. Unlike the earlier Transformer run, the best exact-match result was not an isolated spike: epochs 18 to 20 all remained around 0.967, which suggests that the model is training stably and that the result is reliable.

Conclusion:
The Transformer now appears to be the strongest model tested so far. The next step is to keep the epoch 19 checkpoint as the final Transformer model and evaluate both the LSTM and Transformer on the test set.

## Inspection of final Transformer checkpoint

I inspected 15 development examples from the strongest Transformer checkpoint.

### Overall impression

The Transformer predictions are extremely strong. In this sample, 14 out of 15 predictions were exact. The model handles passive constructions, clausal complements, and ordinary predicate-argument mappings very well.

### Examples of exact predictions

**Source**  
`Liam hoped that a box was burned by a girl .`

**Gold**  
`hope ( agent = Liam , ccomp = burn ( theme = box , agent = girl ) )`

**Prediction**  
`hope ( agent = Liam , ccomp = burn ( theme = box , agent = girl ) )`

This shows correct recovery of nested event structure.

---

**Source**  
`A melon was given to a girl by the guard .`

**Gold**  
`give ( theme = melon , recipient = girl , agent = * guard )`

**Prediction**  
`give ( theme = melon , recipient = girl , agent = * guard )`

This shows correct passive role assignment.

---

**Source**  
`The girl offered the weapon beside a machine to a chicken .`

**Gold**  
`offer ( agent = * girl , theme = * weapon ( nmod . beside = machine ) , recipient = chicken )`

**Prediction**  
`offer ( agent = * girl , theme = * weapon ( nmod . beside = machine ) , recipient = chicken )`

This shows correct modifier structure in a more complex example.

### Remaining error example

**Source**  
`Liam painted a box on a table beside the chair .`

**Gold**  
`paint ( agent = Liam , theme = box ( nmod . on = table ( nmod . beside = * chair ) ) )`

**Prediction**  
The prediction starts correctly but then repeats the modifier structure around `chair` and becomes incomplete.

### Interpretation

The dominant remaining weakness appears to be in harder modifier-attachment cases, where the model can still over-generate or attach phrases incorrectly. However, compared with the LSTM baseline, the Transformer now appears much more reliable overall and produces many fully correct logical forms.