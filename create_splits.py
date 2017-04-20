import numpy as np
import os
import argparse
from shutil import copyfile
from wavenet.audio_reader import find_files

def get_arguments():
    parser = argparse.ArgumentParser(description='Create train and test splits for wavenet data.')
    parser.add_argument('inputdir', type=str,
                        help='The directory where all the wav files are stored.')
    parser.add_argument('outputdir', type=str,
                        help='The directory where all the wav files are stored.')
    parser.add_argument('--testpct', type=float, default=0.1,
                        help='The percentage of files to hold out for testing.')
    return parser.parse_args()

def main():
    args = get_arguments()

    files = find_files(args.inputdir)

    indices = np.arange(len(files))
    np.random.shuffle(indices)

    end_train = int(np.round((1 - args.testpct) * len(files)))
    train_indices = indices[:end_train]
    test_indices = indices[end_train:]

    train_dir = os.path.join(os.path.abspath(args.outputdir), 'train')
    test_dir = os.path.join(os.path.abspath(args.outputdir), 'test')
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    for t in train_indices:
        src = os.path.join(args.inputdir, files[t])
        dest = os.path.join(train_dir, files[t].replace(args.inputdir, '')[1:])
        print src
        print dest
        print os.path.dirname(dest)
        if not os.path.exists(os.path.dirname(dest)):
            os.makedirs(os.path.dirname(dest))
        copyfile(src, dest)

    for t in test_indices:
        src = os.path.join(args.inputdir, files[t])
        dest = os.path.join(test_dir, files[t].replace(args.inputdir, '')[1:])
        if not os.path.exists(os.path.dirname(dest)):
            os.makedirs(os.path.dirname(dest))
        copyfile(src, dest)

if __name__ == '__main__':
    main()
