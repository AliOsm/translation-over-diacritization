import argparse

from os.path import join
from pyarabic.araby import LETTERS as AR_LETTERS
from tqdm import tqdm

def extract_diacritics_list(diac_org_line, DIAC, AR_LETTERS):
    diacritics_list = list()
    diac_tmp = ''
    for char in diac_org_line:
        if char in AR_LETTERS:
            diacritics_list.append(diac_tmp)
            diac_tmp = ''
        if char in DIAC:
            diac_tmp += char
    diacritics_list.append(diac_tmp)
    return diacritics_list[1:]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data_dir')
    args = parser.parse_args()

    DIACS = ''.join(['َ', 'ً', 'ُ', 'ٌ', 'ِ', 'ٍ', 'ّ', 'ْ'])
    AR_LETTERS += 'ى'

    for dataset_split in ['train', 'test']:
        bpe_file_name = 'ar.bpe.%s' % dataset_split
        diac_org_file_name = 'ar-diac.org.%s' % dataset_split
        diac_bpe_file_name = 'ar-diac.bpe.%s' % dataset_split

        with open(join(args.data_dir, bpe_file_name), 'r') as file:
            bpe_lines = file.readlines()
        
        with open(join(args.data_dir, diac_org_file_name), 'r') as file:
            diac_org_lines = file.readlines()

        diac_bpe_lines = list()
        for bpe_line, diac_org_line in tqdm(zip(bpe_lines, diac_org_lines)):
            diacritics_list = extract_diacritics_list(diac_org_line, DIACS, AR_LETTERS)
            diac_bpe_line = ''
            diacritics_list_count = 0
            for char in bpe_line.strip():
                diac_bpe_line += char
                if char in AR_LETTERS:
                    diac_bpe_line += diacritics_list[diacritics_list_count]
                    diacritics_list_count += 1
            diac_bpe_lines.append(diac_bpe_line)

        with open(join(args.data_dir, diac_bpe_file_name), 'w') as file:
            file.write('\n'.join(diac_bpe_lines))
