import random
import torch
import torch.nn as nn


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