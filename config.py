import torch
import torch.nn as nn

class config(object):
    lr = 0.000001
    do_training = False
    load_weights = True
    load_best = True
    regen = False


    ref_time = 3
    pred_time = 3
    x_dim = 10
    points = 10
    criterion = nn.SmoothL1Loss()
    scaler = 'minmax' # must be standard or minmax
    bidirection = True
    batch_size = 512
    n_epochs = 40
    clip = 1
    n_layers = 2
    enc_hid_dim = 128
    dec_hid_dim = 128
    enc_drop = 0.2
    dec_drop = 0.2
    train_ratio = 0.9
    n_test_data = 300
    teacher_forcing_ratio = 0.3


    data_folder = './processed_data'
    all_data_file = './data/data/all_data.csv'
    train_data_file = './data/data/train_data.pkl'
    val_data_file = './data/data/val_data.pkl'
    test_data_file = './data/data/test_data.pkl'
    speed_scaler_file = './data/others/speed_scaler.pkl'
    dist_scaler_file = './data/others/dist_scaler.pkl'
    model_weights = './data/weights/model.pth.tar'
    model_weights_best = './data/weights/model_best.pth.tar'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")