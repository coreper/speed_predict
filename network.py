import torch
import torch.nn as nn
import torch.nn.functional as F
import config
import math
import random
args = config.config()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Encoder(nn.Module):
    def __init__(self, input_size, enc_hid_dim, dec_hid_dim, n_layers, bi=True):
        super().__init__()
        self.bi = bi
        self.rnn = nn.GRU(input_size, enc_hid_dim, num_layers=n_layers, bidirectional=bi)
        if bi:
            self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        else:
            self.fc = nn.Linear(enc_hid_dim, dec_hid_dim)

    def forward(self, src):
        outputs, hidden = self.rnn(src)
        if self.bi:
            hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))
        else:
            hidden = torch.tanh(self.fc(hidden))
        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        src_len = encoder_outputs.shape[0]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, output_size, enc_hid_dim, dec_hid_dim, dropout, attention, emb_dim=64):
        super().__init__()
        self.output_dim = output_size
        self.attention = attention
        self.embedding = nn.Embedding(output_size, emb_dim)
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, 1)
        self.dropout = nn.Dropout(dropout)
    def forward(self, input, hidden, encoder_outputs):
        input = input.unsqueeze(0).long()
        embedded = self.dropout(self.embedding(input))
        a = self.attention(hidden, encoder_outputs)
        a = a.unsqueeze(1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        weighted = torch.bmm(a, encoder_outputs)
        weighted = weighted.permute(1, 0, 2)

        rnn_input = torch.cat((embedded, weighted), dim=2)
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        assert (output == hidden).all()
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))

        return prediction, hidden.squeeze(0)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio):
        encoder_outputs, hidden = self.encoder(src)
        input = src[-1, :, 0]  # the input is the speed of the last point
        if args.do_training:
            outputs = torch.zeros(args.points, args.batch_size, 1).to(self.device)
        else:
            outputs = torch.zeros(args.points, 1, 1).to(self.device)

        if args.do_training:
            for t in range(0, args.points):
                output, hidden = self.decoder(input, hidden, encoder_outputs)
                outputs[t] = output
                teacher_force = random.random() < teacher_forcing_ratio
                input = trg[t] if teacher_force else output.squeeze()
        else:
            for t in range(0, args.points):
                output, hidden = self.decoder(input, hidden, encoder_outputs)
                outputs[t] = output
                input = output.view([1])

        return outputs

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)