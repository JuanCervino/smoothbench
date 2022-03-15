import numpy as np
import argparse
import prettytable
import pandas as pd
import sys
import os
import pickle as pkl
from csv import writer

from smooth.lib import reporting, misc
from smooth import datasets

#TODO(AR): Currently no support for multiple trials

def scrape_results(df, trials, adv, split='Validation'):

    assert split in ['Validation', 'Test']

    all_dfs = []
    for trial in trials:
        trial_df = df[(df['Trial-Seed'] == trial) & (df['Eval-Method'] == adv) \
            & (df.Split == split)]

        # extract the row and epoch with the best performance for given adversary
        best_row = trial_df[trial_df.Accuracy == trial_df.Accuracy.max()]
        best_epoch = best_row.iloc[0]['Epoch']
        best_path = best_row.iloc[0]['Output-Dir']

        best_df = df[(df.Epoch == best_epoch) & (df['Output-Dir'] == best_path) \
            & (df['Trial-Seed'] == trial)]
        all_dfs.append(best_df)

    return pd.concat(all_dfs, ignore_index=True)

def find(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]

if __name__ == '__main__':
    np.set_printoptions(suppress=True)

    parser = argparse.ArgumentParser(description='Collect results')
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--file_to_write', type=str, required=True)
    parser.add_argument('--depth', type=int, default=1, help='Results directories search depth')
    args = parser.parse_args()

    sys.stdout = misc.Tee(os.path.join(args.input_dir, 'results.txt'), 'w')

    records = reporting.load_records(args.input_dir, depth=args.depth)

    eval_methods = records['Eval-Method'].unique()
    dataset_names = records['Dataset'].unique()
    train_algs = records['Train-Alg'].unique()
    trials = records['Trial-Seed'].unique()

    # print(records)

    idx = find(args.input_dir,'_')

    # print(max(records['Accuracy']), args.input_dir.index('_'), args.input_dir[idx[0]+1:idx[1]])

    with open(args.file_to_write+'.csv', 'a') as f_object:
        # Pass this file object to csv.writer()
        # and get a writer object
        writer_object = writer(f_object)

        # Pass the list as an argument into
        # the writerow()
        writer_object.writerow([args.input_dir[idx[0]+1:idx[1]], max(records['Accuracy'])])

        # Close the file object
        f_object.close()


                
                

