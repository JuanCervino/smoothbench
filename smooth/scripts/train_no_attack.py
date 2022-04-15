import argparse
import torch
import os
import json
import pandas as pd
import time
from datetime import datetime
import pickle
import scipy

# Juan Added
import numpy as np
from torch.utils.data import Dataset, Subset, DataLoader, random_split
from torchvision.utils import save_image
from smooth import laplacian
from sklearn.metrics.pairwise import euclidean_distances
import pickle as pkl

from humanfriendly import format_timespan

from sklearn.neighbors import kneighbors_graph


from smooth import datasets
from smooth import algorithms
from smooth import attacks
from smooth import hparams_registry
from smooth.lib import misc, meters
from smooth import laplacian


def main(args, hparams, test_hparams):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hparams['regularizer'] = args.regularizer
    hparams['unlab_batch_size'] = args.unlab_batch_size
    hparams['heat_kernel_t'] = args.heat_kernel_t


    dataset = vars(datasets)[args.dataset](args.data_dir, args.per_labeled)
    # 'train_labeled', 'train_unlabeled', 'train_all', 'test'
    train_lab_ldr, train_unl_ldr, train_all_ldr, test_ldr = datasets.to_loaders(dataset, hparams)

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
        if args.algorithm not in ['ERM_AVG_LIP_RND', 'ERM_AVG_LIP_KNN', 'ERM_LAMBDA_LIP', 'ERM_AVG_LIP_CHEAT','ERM_AVG_LIP_TRANSFORM','ERM_AVG_LIP_CNN_METRIC','ERM_AVG_LIP_KNN_AUG']:
##########################################################
##########################################################
################## ERM  ##################################
##########################################################
##########################################################

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
##########################################################
##########################################################
################## Random Laplacian
##########################################################
##########################################################
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
        ################## KNN LAPLACIAN
        ##########################################################
        ##########################################################
        elif args.algorithm == 'ERM_AVG_LIP_KNN':

            # Here we get the dataset
            if epoch == 0:
                dataset_unlab = vars(datasets)[args.dataset](args.data_dir, 1, transform=args.unlab_augmentation != 1)
                _, train_unl_ldr, train_all_ldr, test_ldr = datasets.to_loaders(dataset_unlab, hparams)
            train_all_ldr_iter = iter (train_all_ldr)
            train_all_ldr_iter_counter = 0
            # dataset_unlab = dataset_unlab.splits['train_all']

            # Now we load the neighbours
            knn = pkl.load(open(args.precalculated_folder + '/knn.p', "rb"))

            for batch_idx, (imgs, labels) in enumerate(train_lab_ldr):
                print(train_all_ldr_iter_counter)
                timer.batch_start()

                # Get the points for the Laplacian
                if epoch % 20 == 0:
                    while train_all_ldr_iter_counter * hparams['unlab_batch_size'] + hparams['unlab_batch_size'] < 50000:
                        algorithm.optimizer.zero_grad()
                        cum = 0
                        train_all_ldr_iter_counter = train_all_ldr_iter_counter+1
                        batch_unlab, _ = next(train_all_ldr_iter)

                        batch_idx_lap_knn = knn[np.arange(train_all_ldr_iter_counter*hparams['unlab_batch_size'],
                                                          train_all_ldr_iter_counter*hparams['unlab_batch_size']+hparams['unlab_batch_size'],
                                                          1, dtype=int)].indices

                        # save_image(batch_unlab[0], 'img1.png')
                        batch_idx_lap_knn = np.array(batch_idx_lap_knn).reshape((hparams['unlab_batch_size'], 10))
                        batch_idx_lap_knn = batch_idx_lap_knn [:, 0 : args.k]

                        batch_idx_lap_knn = np.hstack([np.arange(train_all_ldr_iter_counter * hparams['unlab_batch_size'],
                                                                 train_all_ldr_iter_counter * hparams['unlab_batch_size'] +
                                                                 hparams['unlab_batch_size'],
                                                                 1, dtype=int)[:,np.newaxis], batch_idx_lap_knn])

                        batch_idx_lap_knn = batch_idx_lap_knn.ravel()
                        laplacian_dataloader = torch.utils.data.Subset(dataset_unlab.splits['train_all'],
                                                                       batch_idx_lap_knn)
                        laplacian_ldr = torch.utils.data.DataLoader(laplacian_dataloader,
                                                                    batch_size=args.k+1, shuffle=False,
                                                                    num_workers=12)

                        for batch_idx_lap, (imgs_lap, labels_lap) in enumerate(laplacian_ldr):
                            # central = imgs_lap[0][None]
                            # print(central.shape)
                            imgs_lap = imgs_lap.to(device)
                            cum += torch.sum(torch.nn.functional.softmax(algorithm.predict(imgs_lap[0][None])) * (args.k * torch.nn.functional.softmax(algorithm.predict(imgs_lap[0][None]))
                                                                                                                  - torch.nn.functional.softmax(algorithm.predict(imgs_lap[1::]).sum(dim=0))
                                                                                                                  )
                                             )

                        # We need to take gradients because the memory explotes

                        print('here',train_all_ldr_iter_counter, cum, cum * args.regularizer)
                        cum = args.regularizer * cum
                        cum.backward()
                        algorithm.optimizer.step()
                        algorithm.optimizer.zero_grad()

                # Take CEL step
                # imgs_unlab = imgs_unlab.to(device)
                imgs, labels = imgs.to(device), labels.to(device)

                algorithm.step(imgs, labels)

                if batch_idx % dataset.LOG_INTERVAL == 0:
                    print(f'Train epoch {epoch}/{dataset.N_EPOCHS} ', end='')
                    print(f'[{batch_idx * imgs.size(0)}/{len(train_lab_ldr.dataset)}', end=' ')
                    print(f'({100. * batch_idx / len(train_lab_ldr):.0f}%)]\t', end='')
                    for name, meter in algorithm.meters.items():
                        print(f'{name}: {meter.val:.3f} (avg. {meter.avg:.3f})\t', end='')
                    print(f'Time: {timer.batch_time.val:.3f} (avg. {timer.batch_time.avg:.3f})')

                # train_all_ldr_iter_counter = train_all_ldr_iter_counter + 1
                # if train_all_ldr_iter_counter*args.unlab_batch_size >=50000:
                #     train_all_ldr_iter_counter = 0
                #     train_all_ldr_iter = iter(train_all_ldr_iter)

                timer.batch_end()

##########################################################
##########################################################
################## KNN LAPLACIAN AUGMENTATIONS
##########################################################
##########################################################
        elif args.algorithm == 'ERM_AVG_LIP_KNN_AUG':

            # Here we get the dataset
            dataset_unlab = vars(datasets)[args.dataset](args.data_dir, 1, transform = args.unlab_augmentation!=1)
            dataset_unlab = dataset_unlab.splits['train_all']
            # im, la =  dataset_unlab[0]
            # save_image(im , 'img1.png')
            # _, _, train_all_ldr_unlab, _ = datasets.to_loaders(dataset_unlab, hparams)
            # train_all_ldr_full = DataLoader(dataset.splits['train_all'], batch_size=dataset.splits['train_all'].__len__(), shuffle=False)

            # Now we load the neighbours
            knn = pkl.load(open(args.precalculated_folder + '/knn.p', "rb"))


            for batch_idx, (imgs, labels) in enumerate(train_lab_ldr):

                timer.batch_start()

                # Get the points for the Laplacian
                batch_idx_lap = np.random.choice(len(dataset_unlab), args.unlab_batch_size)


                imgs_unlab = [torch.Tensor() for l in range(args.unlab_batch_size)]
                # ten = torch.Tensor
                for i in range(args.unlab_batch_size):
                    batch_idx_lap_knn = knn[batch_idx_lap[i]].indices.tolist()[0:args.k]
                    laplacian_dataloader = torch.utils.data.Subset(dataset_unlab, batch_idx_lap_knn + [batch_idx_lap[i]])
                    laplacian_ldr = torch.utils.data.DataLoader(laplacian_dataloader, batch_size=laplacian_dataloader.__len__(), shuffle=True,
                                                   num_workers=12)
                    if args.unlab_augmentation == 1:
                        batch_samples_ldr_iterator = iter(laplacian_ldr)  # set shuffle to False
                        # imgs_unlab[i] = next(batch_samples_ldr_iterator)
                        imgs_unlab[i], _ = next(batch_samples_ldr_iterator)
                    else:
                        batch_samples_ldr_iterator = iter(laplacian_ldr)  # set shuffle to False
                        # imgs_unlab[i] = next(batch_samples_ldr_iterator)
                        imgs_unlab[i], _ = next(batch_samples_ldr_iterator)

                        for j in range(args.unlab_augmentation-1):
                            batch_samples_ldr_iterator = iter(laplacian_ldr)
                            # print(type(imgs_unlab[i]))
                            # imgs_unlab[i] = torch.cat((imgs_unlab[i],next(batch_samples_ldr_iterator)))
                            aux, _ = next(batch_samples_ldr_iterator)
                            imgs_unlab[i] = torch.cat((imgs_unlab[i],aux))
                            # print(len(imgs_unlab[i]))
                        # ten = ten.to(device)
                        imgs_unlab[i] = imgs_unlab[i].to(device)
                        # print(i,j,'here',args.unlab_augmentation)

                # imgs_unlab = imgs_unlab.to(device)
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

        class_wise = args.class_wise # To do deal with this
        if class_wise:
            test_clean_classwise_acc = misc.class_wise_accuracy(algorithm, test_ldr, device)
            add_results_row([epoch, test_clean_classwise_acc, 'ERM', 'Test'])

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
        if class_wise:
            print("Clean test classwise accuracy:", test_clean_classwise_acc)  # Juan Changed val. for test. AKS Alex
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

    parser = argparse.ArgumentParser(description='Smoothness')
    parser.add_argument('--data_dir', type=str, default='./smooth/data')
    parser.add_argument('--output_dir', type=str, default='train_output')
    parser.add_argument('--dataset', type=str, default='MNIST', help='Dataset to use')
    parser.add_argument('--algorithm', type=str, default='ERM', help='Algorithm to run')
    # parser.add_argument('--test_attacks', type=str, nargs='+', default=['PGD_Linf'])  # Juan Changed this
    parser.add_argument('--hparams', type=str, help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0, help='Seed for hyperparameters')
    parser.add_argument('--trial_seed', type=int, default=0, help='Trial number')
    parser.add_argument('--seed', type=int, default=0, help='Seed for everything else')
    parser.add_argument('--class_wise', type=bool, default=False, help='compute the class wise accuracy')


    # Juan added this
    parser.add_argument('--normalize', type=bool, default=False, help='Normalize the Laplacian')
    parser.add_argument('--regularizer', type=float, default=.1, help='Regularizer for the SSL')
    parser.add_argument('--per_labeled', type=float, default=1., help='Percentage of training set that will be labeled (between (0,1])')
    parser.add_argument('--unlab_batch_size', type=int, default=128, help='Batchsize used to compute Laplacian')
    parser.add_argument('--heat_kernel_t', type=float, default=1., help='Value of t in the Heat Kernel computation')

    # Laplacian
    parser.add_argument('--k', type=int, default=3, help='Number of KNNs')
    parser.add_argument('--precalculated_folder', type=str, default='None', help='Folder with precalculated KNNs')
    parser.add_argument('--unlab_augmentation', type=int, default=1,
                        help='Number of augmentations in unlab data, if it is larger than 1, data augmentation will be used')

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