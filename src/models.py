import random
import torch
import torch.nn as nn
import math



class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, num_layers=1, dropout=0.2):
        super().__init__()

        self.embedding = nn.Embedding(input_dim, emb_dim)

        self.lstm = nn.LSTM(
            emb_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src: [batch_size, src_len]

        # [batch_size, src_len] -> [batch_size, src_len, emb_dim]
        embedded = self.dropout(self.embedding(src))

        # hidden/cell are what we pass to the decoder
        _, (hidden, cell) = self.lstm(embedded)

        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, num_layers=1, dropout=0.2):
        super().__init__()

        self.output_dim = output_dim

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.lstm = nn.LSTM(
            emb_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_token, hidden, cell):
        # input_token: [batch_size]
        input_token = input_token.unsqueeze(1)  # -> [batch_size, 1]

        embedded = self.dropout(self.embedding(input_token))  # [batch_size, 1, emb_dim]

        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))

        # output is [batch_size, 1, hidden_dim]
        prediction = self.fc_out(output.squeeze(1))  # -> [batch_size, output_dim]

        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        # easiest if encoder and decoder hidden sizes match
        assert encoder.lstm.hidden_size == decoder.lstm.hidden_size
        assert encoder.lstm.num_layers == decoder.lstm.num_layers

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        # src: [batch_size, src_len]
        # tgt: [batch_size, tgt_len]

        batch_size = src.shape[0]
        tgt_len = tgt.shape[1]
        tgt_vocab_size = self.decoder.output_dim

        # store decoder outputs for every time step
        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)

        # encoder reads the source sentence first
        hidden, cell = self.encoder(src)

        # decoder starts from <bos>
        input_token = tgt[:, 0]

        for t in range(1, tgt_len):
            prediction, hidden, cell = self.decoder(input_token, hidden, cell)

            outputs[:, t, :] = prediction

            top1 = prediction.argmax(1)

            # sometimes use the gold token, sometimes use the model prediction
            teacher_force = random.random() < teacher_forcing_ratio
            input_token = tgt[:, t] if teacher_force else top1

        return outputs

    def greedy_decode(self, src, bos_id, eos_id, max_len=50):
        # used at eval time when we don't want teacher forcing

        batch_size = src.shape[0]

        hidden, cell = self.encoder(src)

        input_token = torch.full(
            (batch_size,),
            bos_id,
            dtype=torch.long,
            device=self.device
        )

        decoded_tokens = [input_token]

        # keep track of which sequences have already predicted <eos>
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        for _ in range(max_len - 1):
            prediction, hidden, cell = self.decoder(input_token, hidden, cell)

            next_token = prediction.argmax(1)

            # once a sequence is finished, keep it at <eos>
            next_token = torch.where(finished, torch.full_like(next_token, eos_id), next_token)

            decoded_tokens.append(next_token)

            # update finished mask
            finished = finished | (next_token == eos_id)

            # next decoder input
            input_token = next_token

            # now we can stop if every sequence is finished
            if torch.all(finished):
                break

        decoded_tokens = torch.stack(decoded_tokens, dim=1)
        return decoded_tokens


def build_lstm_seq2seq(
    src_vocab_size,
    tgt_vocab_size,
    device,
    emb_dim=128,
    hidden_dim=256,
    num_layers=1,
    dropout=0.2,
):
    encoder = Encoder(
        input_dim=src_vocab_size,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )

    decoder = Decoder(
        output_dim=tgt_vocab_size,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )

    model = Seq2Seq(encoder, decoder, device).to(device)
    return model


# Transformer model based on Vaswani et al. (2017)
# and the PyTorch nn.Transformer reference implementation.


class TransformerSeq2Seq(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        device,
        emb_dim=128,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=256,
        dropout=0.2,
        max_len=100,
        src_pad_id=0,
        tgt_pad_id=0,
    ):
        super().__init__()

        self.device = device
        self.emb_dim = emb_dim
        self.max_len = max_len
        self.src_pad_id = src_pad_id
        self.tgt_pad_id = tgt_pad_id
        self.tgt_vocab_size = tgt_vocab_size

        # token embeddings
        self.src_tok_emb = nn.Embedding(src_vocab_size, emb_dim)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, emb_dim)

        # learned positional embeddings
        self.src_pos_emb = nn.Embedding(max_len, emb_dim)
        self.tgt_pos_emb = nn.Embedding(max_len, emb_dim)

        self.dropout = nn.Dropout(dropout)

        self.transformer = nn.Transformer(
            d_model=emb_dim,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.fc_out = nn.Linear(emb_dim, tgt_vocab_size)

    def make_src_padding_mask(self, src):
        # True where src has padding
        return src == self.src_pad_id

    def make_tgt_padding_mask(self, tgt):
        # True where tgt has padding
        return tgt == self.tgt_pad_id

    def make_causal_mask(self, seq_len):
        # block attention to future target tokens
        return torch.triu(
            torch.ones(seq_len, seq_len, device=self.device),
            diagonal=1
        ).bool()

    def add_positional_info(self, token_ids, tok_emb, pos_emb):
        # token_ids: [batch_size, seq_len]
        batch_size, seq_len = token_ids.shape

        positions = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, seq_len)

        x = tok_emb(token_ids) * math.sqrt(self.emb_dim)
        x = x + pos_emb(positions)

        return self.dropout(x)

    def forward(self, src, tgt, teacher_forcing_ratio=0.0):
        """
        Keep the same shape convention as the LSTM model.

        src: [batch_size, src_len]
        tgt: [batch_size, tgt_len]

        We use tgt[:, :-1] as decoder input and predict tgt[:, 1:].
        Position 0 in the returned tensor is left as zeros so train.py
        can keep using outputs[:, 1:, :].
        """
        batch_size, tgt_len = tgt.shape

        outputs = torch.zeros(batch_size, tgt_len, self.tgt_vocab_size, device=self.device)

        src_padding_mask = self.make_src_padding_mask(src)

        # decoder sees everything except the last gold token
        tgt_input = tgt[:, :-1]

        tgt_padding_mask = self.make_tgt_padding_mask(tgt_input)
        causal_mask = self.make_causal_mask(tgt_input.shape[1])

        src_emb = self.add_positional_info(src, self.src_tok_emb, self.src_pos_emb)
        tgt_emb = self.add_positional_info(tgt_input, self.tgt_tok_emb, self.tgt_pos_emb)

        transformer_out = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask,
            tgt_mask=causal_mask,
        )

        logits = self.fc_out(transformer_out)

        # keep same output layout as LSTM code
        outputs[:, 1:, :] = logits

        return outputs

    def encode(self, src):
        src_padding_mask = self.make_src_padding_mask(src)
        src_emb = self.add_positional_info(src, self.src_tok_emb, self.src_pos_emb)

        memory = self.transformer.encoder(
            src_emb,
            src_key_padding_mask=src_padding_mask,
        )

        return memory, src_padding_mask

    def greedy_decode(self, src, bos_id, eos_id, max_len=50):
        batch_size = src.shape[0]

        memory, src_padding_mask = self.encode(src)

        # start every sequence with <bos>
        decoded = torch.full(
            (batch_size, 1),
            bos_id,
            dtype=torch.long,
            device=self.device,
        )

        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        for _ in range(max_len - 1):
            tgt_padding_mask = self.make_tgt_padding_mask(decoded)
            causal_mask = self.make_causal_mask(decoded.shape[1])

            tgt_emb = self.add_positional_info(decoded, self.tgt_tok_emb, self.tgt_pos_emb)

            decoder_out = self.transformer.decoder(
                tgt=tgt_emb,
                memory=memory,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=tgt_padding_mask,
                memory_key_padding_mask=src_padding_mask,
            )

            # only use the newest decoder step
            logits = self.fc_out(decoder_out[:, -1, :])
            next_token = logits.argmax(dim=1)

            # once a sequence ends, keep it at eos
            next_token = torch.where(
                finished,
                torch.full_like(next_token, eos_id),
                next_token,
            )

            decoded = torch.cat([decoded, next_token.unsqueeze(1)], dim=1)

            finished = finished | (next_token == eos_id)

            if torch.all(finished):
                break

        return decoded


def build_transformer_seq2seq(
    src_vocab_size,
    tgt_vocab_size,
    device,
    src_pad_id,
    tgt_pad_id,
    emb_dim=128,
    nhead=4,
    num_encoder_layers=2,
    num_decoder_layers=2,
    dim_feedforward=256,
    dropout=0.2,
    max_len=100,
):
    model = TransformerSeq2Seq(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        device=device,
        emb_dim=emb_dim,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        max_len=max_len,
        src_pad_id=src_pad_id,
        tgt_pad_id=tgt_pad_id,
    ).to(device)

    return model