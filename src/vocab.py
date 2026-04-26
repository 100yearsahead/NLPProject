from collections import Counter


class Vocab:
    """
    Small vocabulary class for mapping tokens to integer ids and back.

    For this project, both the input sentence and the target logical form
    need to be converted into sequences of ids before they can be passed
    into an LSTM or Transformer model.
    """

    def __init__(self, specials=None):
        """
        Start the vocabulary with a few special tokens.

        <pad> is used when batching sequences of different lengths.
        <bos> marks the beginning of a sequence.
        <eos> marks the end of a sequence.
        <unk> is used for tokens that were not seen in training.
        """
        if specials is None:
            specials = ["<pad>", "<bos>", "<eos>", "<unk>"]

        self.token_to_id = {}
        self.id_to_token = {}

        for token in specials:
            self.add_token(token)

        self.pad_token = "<pad>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"

        self.pad_id = self.token_to_id[self.pad_token]
        self.bos_id = self.token_to_id[self.bos_token]
        self.eos_id = self.token_to_id[self.eos_token]
        self.unk_id = self.token_to_id[self.unk_token]

    def add_token(self, token):
        """
        Add a token if it is not already in the vocabulary.
        """
        if token not in self.token_to_id:
            idx = len(self.token_to_id)
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token

    def build_from_texts(self, texts, min_freq=1):
        """
        Build the vocabulary from a collection of texts.

        I am using simple whitespace tokenisation here because COGS is already
        a clean, structured dataset, and the punctuation inside the logical
        forms is meaningful for the task.
        """
        counter = Counter()

        for text in texts:
            tokens = text.split()
            counter.update(tokens)

        for token, freq in counter.items():
            if freq >= min_freq:
                self.add_token(token)

    def encode(self, text, add_bos=False, add_eos=False):
        """
        Convert a text string into a list of token ids.
        """
        tokens = text.split()
        ids = []

        if add_bos:
            ids.append(self.bos_id)

        for token in tokens:
            ids.append(self.token_to_id.get(token, self.unk_id))

        if add_eos:
            ids.append(self.eos_id)

        return ids

    def decode(self, ids, remove_special=True):
        """
        Convert a list of ids back into a text string.

        This is useful later when checking model predictions.
        """
        tokens = []
        special_tokens = {self.pad_token, self.bos_token, self.eos_token}

        for idx in ids:
            token = self.id_to_token.get(int(idx), self.unk_token)

            if remove_special and token in special_tokens:
                continue

            tokens.append(token)

        return " ".join(tokens)

    def __len__(self):
        """
        Return the size of the vocabulary.
        """
        return len(self.token_to_id)