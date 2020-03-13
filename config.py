import torch
import torch.nn as nn

class config(object):
    data_folder = './Processed_data'
    train_data_file = './data/train_data'
    val_data_file = './data/val_data'
    test_data_file = './data/test_data'
    encoder_weights = './weights/encoder.pth.tar'
    decoder_weights = './weights/decoder.pth.tar'
    encoder_weights_best = './weights/encoder_best.pth.tar'
    decoder_weights_best = './weights/decoder_best.pth.tar'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    ref_time = 3
    pred_time = 3
    hidden_size = 128
    teacher_forcing_ratio = 0.5
    rnn_layers = 2
    lr = 0.000001
    x_dim = 8
    points = 10 # points should smaller than 10*pred_time
    max_len = max(ref_time, pred_time) * 10
    criterion = nn.SmoothL1Loss()
    use_degrade_loss = False
    print_every = 3000
    num_val_sample = 300
    train_ratio = 0.9


    do_training = True
    load_weights = False
    load_best = False
    regen = True