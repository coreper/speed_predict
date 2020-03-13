import numpy as np
import math
import time
import random
import shutil
import torch
from torch import optim
import matplotlib.pyplot as plt
import config
args = config.config()
from sklearn.metrics import mean_absolute_error

def train(input_tensor, target_tensor, encoder, decoder,
          encoder_optimizer, decoder_optimizer, criterion,
          max_length, device, teacher_forcing_ratio):
    encoder.train()
    decoder.train()
    encoder_hidden = encoder.initHidden()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = input_tensor[-1, 1].view(1, -1)
    decoder_hidden = encoder_hidden
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)

            decoder_input = target_tensor[di]
            if args.use_degrade_loss:
                loss += criterion(decoder_output.squeeze(), target_tensor[di])/(di/10 + 1) * 50000
            else:
                loss += criterion(decoder_output.squeeze(), target_tensor[di]) * 10000
    else:
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)

            decoder_input = decoder_output
            if args.use_degrade_loss:
                loss += criterion(decoder_output.squeeze(), target_tensor[di])/(di/10 + 1) * 50000
            else:
                loss += criterion(decoder_output.squeeze(), target_tensor[di]) * 10000

    loss.backward()
    torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1)
    torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1)
    encoder_optimizer.step()
    decoder_optimizer.step()
    return loss.item() / target_length


def evaluate(input_tensor, target_tensor, encoder, decoder,
             criterion, max_length, device):
    encoder.eval()
    decoder.eval()
    input_length = input_tensor.size()[0]
    target_length = target_tensor.size(0)
    with torch.no_grad():
        encoder_hidden = encoder.initHidden()
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
        val_loss = 0

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = input_tensor[-1, 1].view(1, -1)
        decoder_hidden = encoder_hidden
        predict_speed = []
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_input = decoder_output.detach()
            if args.use_degrade_loss:
                val_loss += criterion(decoder_output.squeeze(), target_tensor[di])/(di/10 + 1) * 50000
            else:
                val_loss += criterion(decoder_output.squeeze(), target_tensor[di]) * 10000
            predict_speed.append(decoder_output.item())

        return predict_speed, val_loss.item() / target_length


def trainIters(encoder, decoder, learning_rate, training_data, val_data, print_every, device,
               encoder_weights, decoder_weights, criterion, max_length, teacher_forcing_ratio, speed_scaler):
    start = time.time()
    best_prediction = 1
    print_loss_total = 0
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

    for iter in range(1, len(training_data)):
        training_pair = training_data[iter]
        input_tensor = torch.Tensor(training_pair[0]).to(device)
        target_tensor = torch.Tensor(training_pair[1]).to(device)

        loss = train(
            input_tensor=input_tensor,
            target_tensor=target_tensor,
            encoder=encoder,
            decoder=decoder,
            encoder_optimizer=encoder_optimizer,
            decoder_optimizer=decoder_optimizer,
            criterion=criterion,
            max_length=max_length,
            device=device,
            teacher_forcing_ratio=teacher_forcing_ratio,

        )
        print_loss_total += loss

        if iter % print_every == 0:
            # save the model
            torch.save({'state_dict': encoder.state_dict()}, encoder_weights)
            torch.save({'state_dict': decoder.state_dict()}, decoder_weights)

            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('Total time: %s \nIterations: %d (%d%%) \nTraining loss: %.5f' % (
            timeSince(start, iter / len(training_data)),
            iter, iter / len(training_data) * 100, print_loss_avg))

            val_pairs = [random.choice(val_data) for i in range(args.num_val_sample)] # evaluate sample (300) data
            # val_pairs = val_data # evaluate all data
            val_total_loss = 0
            val_speed_errors = []
            for val_iter in range(len(val_pairs)):
                val_pair = val_pairs[val_iter]
                val_input_tensor = torch.Tensor(val_pair[0]).to(device)
                val_target_tensor = torch.Tensor(val_pair[1]).to(device)


                pred_speed, val_loss = evaluate(
                    input_tensor=val_input_tensor,
                    target_tensor=val_target_tensor,
                    encoder=encoder,
                    decoder=decoder,
                    criterion=criterion,
                    max_length=max_length,
                    device=device,
                )
                val_total_loss += val_loss

                predict_speed_real = speed_scaler.inverse_transform(np.array(pred_speed).reshape(-1, 1))
                val_target_real = speed_scaler.inverse_transform(val_pair[1].reshape(-1, 1))

                real_error = mean_absolute_error(predict_speed_real, val_target_real)
                val_speed_errors.append(real_error.item())

            mean_speed_error = np.array(val_speed_errors).mean()
            print('Evaluate loss: {}'.format(round(val_total_loss / len(val_pairs), 3)))
            print('Real speed error {} mile/hr'.format(round(mean_speed_error*2.23694, 3))) # meter/sec to mile/hour


            if mean_speed_error < best_prediction:
                shutil.copyfile(args.encoder_weights, args.encoder_weights_best)
                shutil.copyfile(args.decoder_weights, args.decoder_weights_best)
                best_prediction = mean_speed_error
                print('\33[93mBest weights updated\33[0m')
            print('-------------------------------')



def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def show_predictiopn(test_data, dist_scaler, speed_scaler, encoder, decoder, criterion, max_length, device):
    if not args.load_weights:
        print('Set load weights in config')
        exit()
    # prepare for the text data
    for instance_df in test_data:
        if instance_df.shape[0] <= (args.ref_time + args.pred_time) * 10:
            continue
        instance_np = instance_df.values
        x_np_test = instance_np[:, :args.x_dim]
        speed_test = instance_np[:, 1]
        dist_test = instance_np[:, 0]
        rand_position = random.randint(args.ref_time * 10, len(x_np_test) - args.pred_time * 10)

        dist_traj = dist_scaler.inverse_transform(dist_test.reshape(-1, 1))
        dist_traj = np.squeeze(dist_traj)
        speed_traj = speed_scaler.inverse_transform(speed_test.reshape(-1, 1))
        speed_traj = np.squeeze(speed_traj)


        dist_select_gt = dist_scaler.inverse_transform(
            dist_test[rand_position:rand_position + args.pred_time * 10].reshape(-1, 1))
        dist_select_gt = np.squeeze(dist_select_gt)
        dist_select_gt = dist_select_gt[::int(args.pred_time * 10 / args.points)]

        dist_select_ref = dist_scaler.inverse_transform(
            dist_test[rand_position - args.ref_time * 10:rand_position].reshape(-1, 1))
        dist_select_ref = np.squeeze(dist_select_ref)
        speed_select_ref = speed_test[rand_position - args.ref_time * 10:rand_position]
        speed_select_ref = speed_scaler.inverse_transform(speed_select_ref.reshape(-1, 1))


        x_test = x_np_test[rand_position - args.ref_time * 10:rand_position, :]
        y_test = speed_test[rand_position:rand_position + args.pred_time * 10]
        ground_truth = speed_scaler.inverse_transform(y_test.reshape(-1, 1))
        ground_truth = np.squeeze(ground_truth)
        ground_truth = ground_truth[::int(args.pred_time * 10 / args.points)]

        test_input_tensor = torch.Tensor(x_test).to(args.device)
        test_target_tensor = torch.Tensor(ground_truth).to(args.device)

        pred_speed, _ = evaluate(
            input_tensor=test_input_tensor,
            target_tensor=test_target_tensor,
            encoder=encoder,
            decoder=decoder,
            criterion=criterion,
            max_length=max_length,
            device=device,
        )
        predict_speed_real = speed_scaler.inverse_transform(np.array(pred_speed).reshape(-1, 1))

        plt.figure()
        plt.xlabel('Distance to stop line (meter)')
        plt.ylabel('Vehicle speed (m/s)')
        plt.grid(True)
        plt.title('Speed prediction')
        plt.plot(dist_traj, speed_traj, c='y', linestyle='dashed', label='trajectory')
        plt.plot(dist_select_ref, speed_select_ref, c='g', linestyle='dashed', label='Reference points')
        plt.scatter(dist_select_gt, ground_truth, c='g', label='Ground truth')
        plt.plot(dist_select_gt, predict_speed_real, c='r', label='Prediction')
        plt.legend()
        plt.show()