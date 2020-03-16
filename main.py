import pandas as pd
import os
import torch
import data
import pickle
import random
import numpy as np
import torch.utils.data
import network
import time
import train
import config
args = config.config()


# SEED = 806
# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)
# torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
print('- Calculating using {}'.format(args.device))
print('- Predict Horizon {} Seconds'.format(args.pred_time))

if os.path.exists(args.train_data_file) and os.path.exists(args.val_data_file) \
        and os.path.exists(args.test_data_file) and args.regen==False:
    print('- Loading previous train/val/test data')
    with open(args.train_data_file, 'rb') as fp:
        train_data = pickle.load(fp)
    with open(args.val_data_file, 'rb') as fp:
        val_data = pickle.load(fp)
    with open(args.test_data_file, 'rb') as fp:
        test_data = pickle.load(fp)
    df = pd.read_csv(args.all_data_file, index_col=False)

else:
    print('- NEW TRAINING: regenerating the train/val/test data')
    df = pd.DataFrame()
    if not os.path.exists(args.all_data_file) and args.do_training:  # save all the data to a .csv file
        for file in os.listdir(args.data_folder):
            if file.endswith(".csv"):
                data_file = os.path.join(args.data_folder, file)
                df_ = pd.read_csv(data_file, index_col=False)
                df = pd.concat([df, df_], ignore_index=True)
        print('- Creating all data.csv file')
        df.to_csv('./all_data.csv', index=False)
    else:
        print('- Reading all data.csv file')
        df = pd.read_csv(args.all_data_file, index_col=False)

    train_data, val_data, test_data = data.prep_data(
        df=df,
        train_data_file=args.train_data_file,
        val_data_file=args.val_data_file,
        test_data_file=args.test_data_file,
        ref_time=args.ref_time,
        pred_time=args.pred_time
    )

_, _, _, speed_scaler, dist_scaler, other_scaler = data.scale_data(df)

if args.do_training:
    print('- Total {:,} and {:,} training and validation trajectories'.format(len(train_data), len(val_data)))

train_iterator = torch.utils.data.DataLoader(dataset=train_data,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             num_workers=8,
                                             drop_last=True)

val_iterator = torch.utils.data.DataLoader(dataset=val_data,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           num_workers=8,
                                           drop_last=True)


enc = network.Encoder(input_size=args.x_dim,
                      enc_hid_dim=args.enc_hid_dim,
                      n_layers=args.n_layers,
                      bi=args.bidirection,
                      dec_hid_dim=128)

attn = network.Attention(enc_hid_dim=args.enc_hid_dim,
                         dec_hid_dim=args.dec_hid_dim)

dec = network.Decoder(output_size=args.points,
                      enc_hid_dim = args.enc_hid_dim,
                      dec_hid_dim = args.dec_hid_dim,
                      dropout=args.dec_drop,
                      attention=attn)

model = network.Seq2Seq(enc, dec, args.device).to(args.device)
print(f'- The model has {network.count_parameters(model):,} trainable parameters')
model.apply(network.initialize_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
criterion = args.criterion

# continue training
if args.load_weights and os.path.isfile(args.model_weights):
    try:
        if args.load_best:
            check_encoder = torch.load(args.model_weights_best)
            print('- BEST checkpoints loaded.')
        else:
            check_encoder = torch.load(args.model_weights)
            print('- LAST checkpoints loaded.')
        model.load_state_dict(check_encoder)
    except FileNotFoundError:
        print('Weights file not exist')
        exit()

if args.do_training:
    print('\33[95mTraining start!\33[0m')
    best_valid_loss = float('inf')
    for epoch in range(args.n_epochs):
        start_time = time.time()
        train_loss = train.train(model, train_iterator, optimizer, criterion, args.clip)
        valid_loss = train.evaluate(model, val_iterator, criterion)
        end_time = time.time()

        torch.save(model.state_dict(), args.model_weights)
        epoch_mins, epoch_secs = train.epoch_time(start_time, end_time)
        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Val. Loss: {valid_loss:.3f}')

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), args.model_weights_best)
            print('\t\33[93mBest weights updated\33[0m')
        print('------------------------')

else:
    train.show_predictiopn(test_data=test_data,
                             model=model,
                             speed_scaler = speed_scaler,
                             dist_scaler = dist_scaler)

