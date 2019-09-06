import argparse

from os.path import join
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data_dir')
    args = parser.parse_args()

    for dataset_split in ['train', 'test']:
        with open(join(args.data_dir, 'ar.org.%s' % dataset_split), 'r') as file:
            ar_org_lines = file.readlines()

        with open(join(args.data_dir, 'ar.bpe.%s' % dataset_split), 'r') as file:
            ar_bpe_lines = file.readlines()

        with open(join(args.data_dir, 'en.org.%s' % dataset_split), 'r') as file:
            en_org_lines = file.readlines()

        with open(join(args.data_dir, 'en.bpe.%s' % dataset_split), 'r') as file:
            en_bpe_lines = file.readlines()

        lines = zip(ar_org_lines, ar_bpe_lines, en_org_lines, en_bpe_lines)

        ar_org_lines = list()
        ar_bpe_lines = list()
        en_org_lines = list()
        en_bpe_lines = list()

        for line in lines:
            if len(line[1].split()) > 50 or len(line[3].split()) > 50:
                continue

            ar_org_lines.append(line[0].strip())
            ar_bpe_lines.append(line[1].strip())
            en_org_lines.append(line[2].strip())
            en_bpe_lines.append(line[3].strip())

        with open(join(args.data_dir, 'ar.org.%s' % dataset_split), 'w') as file:
            file.write('\n'.join(ar_org_lines))

        with open(join(args.data_dir, 'ar.bpe.%s' % dataset_split), 'w') as file:
            file.write('\n'.join(ar_bpe_lines))

        with open(join(args.data_dir, 'en.org.%s' % dataset_split), 'w') as file:
            file.write('\n'.join(en_org_lines))

        with open(join(args.data_dir, 'en.bpe.%s' % dataset_split), 'w') as file:
            file.write('\n'.join(en_bpe_lines))

    print("Done!")