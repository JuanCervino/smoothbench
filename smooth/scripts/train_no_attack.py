import argparse
import torch
import os
import json
import pandas as pd
import time
from datetime import datetime

# Juan Added
import numpy as np
from torch.utils.data import Dataset, Subset, DataLoader, random_split
from smooth import laplacian


from humanfriendly import format_timespan



from smooth import datasets
from smooth import algorithms
from smooth import attacks
from smooth import hparams_registry
from smooth.lib import misc, meters
from smooth import laplacian


def main(args, hparams, test_hparams):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hparams['regularizer'] = args.regularizer
    hparams['unlabeled_batch_size'] = args.unlabeled_batch_size
    hparams['heat_kernel_t'] = args.heat_kernel_t


    dataset = vars(datasets)[args.dataset](args.data_dir, args.per_labeled)
    # 'train_labeled', 'train_unlabeled', 'train_all', 'test'
    train_lab_ldr, train_unl_ldr, train_all_ldr, test_ldr = datasets.to_loaders(dataset, hparams)
    print(len(train_lab_ldr),len(train_all_ldr))

    algorithm = vars(algorithms)[args.algorithm](
        dataset.INPUT_SHAPE, 
        dataset.NUM_CLASSES,
        hparams,
        device).to(device)

    adjust_lr = None if dataset.HAS_LR_SCHEDULE is False else dataset.adjust_lr

    # test_attacks = {
    #     a: vars(attacks)[a](algorithm.classifier, test_hparams, device) for a in args.test_attacks}
    #
    columns = ['Epoch', 'Accuracy', 'Eval-Method', 'Split', 'Train-Alg', 'Dataset', 'Trial-Seed', 'Output-Dir']
    results_df = pd.DataFrame(columns=columns)
    def add_results_row(data):
        defaults = [args.algorithm, args.dataset, args.trial_seed, args.output_dir]
        results_df.loc[len(results_df)] = data + defaults

    total_time = 0
    # Juan Added this
    # print(dataset.splits.items(), dataset.splits['train_all'])
    lambdas = np.ones(len(dataset.splits['train_all']))/len(dataset.splits['train_all'])
    print(len(dataset.splits['train_all']),len(lambdas))

    for epoch in range(0, dataset.N_EPOCHS):

        if adjust_lr is not None:
            adjust_lr(algorithm.optimizer, epoch, hparams)

        timer = meters.TimeMeter()
        epoch_start = time.time()

        # Juan added this
        if args.algorithm not in ['ERM_AVG_LIP_RND', 'ERM_AVG_LIP_KNN', 'ERM_LAMBDA_LIP']:
            for batch_idx, (imgs, labels) in enumerate(train_lab_ldr):

                timer.batch_start()
                imgs, labels = imgs.to(device), labels.to(device)
                algorithm.step(imgs, labels)

                if batch_idx % dataset.LOG_INTERVAL == 0:
                    print(f'Train epoch {epoch}/{dataset.N_EPOCHS} ', end='')
                    print(f'[{batch_idx * imgs.size(0)}/{len(train_lab_ldr.dataset)}', end=' ')
                    print(f'({100. * batch_idx / len(train_lab_ldr):.0f}%)]\t', end='')
                    for name, meter in algorithm.meters.items():
                        print(f'{name}: {meter.val:.3f} (avg. {meter.avg:.3f})\t', end='')
                    print(f'Time: {timer.batch_time.val:.3f} (avg. {timer.batch_time.avg:.3f})')

                timer.batch_end()
        elif args.algorithm == 'ERM_AVG_LIP_RND':
            train_all_ldr_iterator = iter(train_all_ldr)
            for batch_idx, (imgs, labels) in enumerate(train_lab_ldr):

                timer.batch_start()
                imgs_unlab, _ = next(train_all_ldr_iterator)
                imgs_unlab = imgs_unlab.to(device)
                imgs, labels = imgs.to(device), labels.to(device)

                algorithm.step(imgs, labels, imgs_unlab)

                if batch_idx % dataset.LOG_INTERVAL == 0:
                    print(f'Train epoch {epoch}/{dataset.N_EPOCHS} ', end='')
                    print(f'[{batch_idx * imgs.size(0)}/{len(train_lab_ldr.dataset)}', end=' ')
                    print(f'({100. * batch_idx / len(train_lab_ldr):.0f}%)]\t', end='')
                    for name, meter in algorithm.meters.items():
                        print(f'{name}: {meter.val:.3f} (avg. {meter.avg:.3f})\t', end='')
                    print(f'Time: {timer.batch_time.val:.3f} (avg. {timer.batch_time.avg:.3f})')

                timer.batch_end()
##########################################################
##########################################################
################## LAMBDA LAPLACIAN
##########################################################
##########################################################
        elif args.algorithm == 'ERM_LAMBDA_LIP':

            # train_all_ldr_iterator = iter(train_all_ldr)
            train_all_ldr_full = DataLoader(dataset.splits['train_all'], batch_size=dataset.splits['train_all'].__len__(), shuffle=False)
            train_all_ldr_full_iter = iter(train_all_ldr_full)
            dataset_unlab, _ = next(train_all_ldr_full_iter)
            # dataset_unlab = dataset_unlab.to(device)
            L = laplacian.get_laplacian(dataset_unlab)

            for batch_idx, (imgs, labels) in enumerate(train_lab_ldr):
                timer.batch_start()
                # Sample according to Lambda
                samples = np.random.choice(len(lambdas), hparams['unlabeled_batch_size'], p=lambdas)

                # samples_lst = [i for i in samples]
                subset = torch.utils.data.Subset(dataset.splits['train_all'], samples)
                batchUnlabeledLambda = DataLoader(subset, batch_size=len(lambdas), shuffle=False)  # set shuffle to False
                train_all_ldr_iterator = iter(batchUnlabeledLambda)
                # print(len(batchUnlabeledLambda))
                imgs_unlab, _ = next(train_all_ldr_iterator)
                imgs_unlab = imgs_unlab.to(device)
                imgs, labels = imgs.to(device), labels.to(device)

                algorithm.step(imgs, labels, imgs_unlab, torch.tensor(lambdas[samples]).to(device).float())

                if batch_idx % dataset.LOG_INTERVAL == 0:
                    print(f'Train epoch {epoch}/{dataset.N_EPOCHS} ', end='')
                    print(f'[{batch_idx * imgs.size(0)}/{len(train_lab_ldr.dataset)}', end=' ')
                    print(f'({100. * batch_idx / len(train_lab_ldr):.0f}%)]\t', end='')
                    for name, meter in algorithm.meters.items():
                        print(f'{name}: {meter.val:.3f} (avg. {meter.avg:.3f})\t', end='')
                    print(f'Time: {timer.batch_time.val:.3f} (avg. {timer.batch_time.avg:.3f})')

                timer.batch_end()

            # Epoch end
            # Update Lambdas

        # save clean accuracies on validation/test sets
        test_clean_acc = misc.accuracy(algorithm, test_ldr, device)
        add_results_row([epoch, test_clean_acc, 'ERM', 'Test'])

        # # save adversarial accuracies on validation/test sets
        # test_adv_accs = []
        # for attack_name, attack in test_attacks.items():
        #     test_adv_acc = misc.adv_accuracy(algorithm, test_ldr, device, attack)
        #     add_results_row([epoch, test_adv_acc, attack_name, 'Test'])
        #     test_adv_accs.append(test_adv_acc)

        epoch_end = time.time()
        total_time += epoch_end - epoch_start

        # print results
        print(f'Epoch: {epoch+1}/{dataset.N_EPOCHS}\t', end='')
        print(f'Epoch time: {format_timespan(epoch_end - epoch_start)}\t', end='')
        print(f'Total time: {format_timespan(total_time)}\t', end='')
        print(f'Training alg: {args.algorithm}\t', end='')
        print(f'Dataset: {args.dataset}\t', end='')
        print(f'Path: {args.output_dir}')
        for name, meter in algorithm.meters.items():
            print(f'Avg. train {name}: {meter.avg:.3f}\t', end='')
        print(f'\nClean test accuracy: {test_clean_acc:.3f}\t', end='') # Juan Changed val. for test. AKS Alex
        # for attack_name, acc in zip(test_attacks.keys(), test_adv_accs):
        #     print(f'{attack_name} val. accuracy: {acc:.3f}\t', end='')
        print('\n')

        # save results dataframe to file
        results_df.to_pickle(os.path.join(args.output_dir, 'results.pkl'))

        # reset all meters
        meters_df = algorithm.meters_to_df(epoch)
        meters_df.to_pickle(os.path.join(args.output_dir, 'meters.pkl'))
        algorithm.reset_meters()

    torch.save(
        {'model': algorithm.state_dict()}, 
        os.path.join(args.output_dir, f'ckpt.pkl'))

    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Adversarial robustness evaluation')
    parser.add_argument('--data_dir', type=str, default='./smooth/data')
    parser.add_argument('--output_dir', type=str, default='train_output')
    parser.add_argument('--dataset', type=str, default='MNIST', help='Dataset to use')
    parser.add_argument('--algorithm', type=str, default='ERM', help='Algorithm to run')
    # parser.add_argument('--test_attacks', type=str, nargs='+', default=['PGD_Linf'])  # Juan Changed this
    parser.add_argument('--hparams', type=str, help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0, help='Seed for hyperparameters')
    parser.add_argument('--trial_seed', type=int, default=0, help='Trial number')
    parser.add_argument('--seed', type=int, default=0, help='Seed for everything else')

    # Juan added this
    parser.add_argument('--normalize', type=bool, default=False, help='Normalize the Laplacian')
    parser.add_argument('--regularizer', type=float, default=.1, help='Regularizer for the SSL')
    parser.add_argument('--per_labeled', type=float, default=1., help='Percentage of training set that will be labeled (between (0,1])')
    parser.add_argument('--unlabeled_batch_size', type=int, default=128, help='Batchsize used to compute Laplacian')
    parser.add_argument('--heat_kernel_t', type=float, default=1., help='Value of t in the Heat Kernel computation ')



    args = parser.parse_args()

    args.output_dir = args.output_dir + '/' + str(args.algorithm) + '_' + str(args.per_labeled) + '_' + datetime.now().strftime("%Y-%m%d-%H%M%S")
    os.makedirs(os.path.join(args.output_dir), exist_ok=True)

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print(f'\t{k}: {v}')

    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    if args.dataset not in vars(datasets):
        raise NotImplementedError(f'Dataset {args.dataset} is not implemented.')

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        seed = misc.seed_hash(args.hparams_seed, args.trial_seed)
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset, seed)

    print ('Hparams:')
    for k, v in sorted(hparams.items()):
        print(f'\t{k}: {v}')

    with open(os.path.join(args.output_dir, 'hparams.json'), 'w') as f:
        json.dump(hparams, f, indent=2)

    test_hparams = hparams_registry.test_hparams(args.algorithm, args.dataset)

    print('Test hparams:')
    for k, v in sorted(test_hparams.items()):
        print(f'\t{k}: {v}')

    with open(os.path.join(args.output_dir, 'test_hparams.json'), 'w') as f:
        json.dump(test_hparams, f, indent=2)

    main(args, hparams, test_hparams)