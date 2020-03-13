import pandas as pd
import os
import torch
import config
args = config.config()
import data
import model
import train

if __name__ == '__main__':
    print('\33[93m- Sequence to Sequence Speed Prediction\n- Predict Horizon {} Seconds\33[0m'.format(args.pred_time))
    print('- Calculating using {}'.format(args.device))

    df = pd.DataFrame()
    if not os.path.exists('./all_data.csv') and args.do_training: # save all the data to a .csv file
        for file in os.listdir(args.data_folder):
            if file.endswith(".csv"):
                data_file = os.path.join(args.data_folder, file)
                df_ = pd.read_csv(data_file, index_col=False)
                df = pd.concat([df, df_], ignore_index=True)
        print('- Creating all data.csv file')
        df.to_csv('./all_data.csv', index=False)
    else:
        print('- Reading all data.csv file')
        df = pd.read_csv('./all_data.csv', index_col=False)


    train_data, val_data, test_data, speed_scaler, dist_scaler = data.prep_data(df)
    if args.do_training:
        print('- Total {:,} training trajectories'.format(len(train_data)))

    encoder = model.EncoderRNN(hidden_size=args.hidden_size, n_layers=args.rnn_layers,
                               input_size=args.x_dim, device=args.device).to(args.device)
    decoder = model.AttnDecoderRNN(hidden_size=args.hidden_size, output_size=1,
                                   n_layers=args.rnn_layers, device=args.device,
                                   max_length=args.max_len).to(args.device)

    print(f'- The model has {(model.count_parameters(encoder) + model.count_parameters(decoder)):,} trainable parameters')


    if args.load_weights and os.path.isfile(args.encoder_weights):
        try:
            if args.load_best:
                check_encoder = torch.load(args.encoder_weights_best)
                check_decoder = torch.load(args.decoder_weights_best)
                print('- BEST checkpoints loaded.')
            else:
                check_encoder = torch.load(args.encoder_weights)
                check_decoder = torch.load(args.decoder_weights)
                print('- LAST checkpoints loaded.')
            encoder.load_state_dict(check_encoder['state_dict'])
            decoder.load_state_dict(check_decoder['state_dict'])
        except FileNotFoundError:
            print('Weights file not exist')
            exit()

    if args.do_training:
        print('\33[95mTraining start!\33[0m')
        train.trainIters(
            encoder=encoder,
            decoder=decoder,
            learning_rate=args.lr,
            training_data=train_data,
            val_data=val_data,
            print_every=args.print_every,
            device=args.device,
            encoder_weights=args.encoder_weights,
            decoder_weights=args.decoder_weights,
            criterion=args.criterion,
            max_length=args.max_len,
            teacher_forcing_ratio=args.teacher_forcing_ratio,
            speed_scaler=speed_scaler
        )

    else: # show some prediction results
        train.show_predictiopn(test_data=test_data,
                               dist_scaler=dist_scaler,
                               speed_scaler=speed_scaler,
                               encoder=encoder,
                               decoder=decoder,
                               criterion=args.criterion,
                               max_length=args.max_len,
                               device=args.device)