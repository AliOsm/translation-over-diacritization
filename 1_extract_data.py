import re
import random
import argparse
import xml.etree.ElementTree as ET

from os import sep
from os import walk
from os.path import join
from string import punctuation as PUNCS
from string import ascii_lowercase as EN_LETTERS
from pyarabic.araby import LETTERS as AR_LETTERS
from tqdm import tqdm

def is_url_exist(text):
    url = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    return len(url) != 0

def remove_characters(text, characters_list):
    return text.translate(str.maketrans('', '', ''.join(characters_list)))

def existance_percentage(text, characters_list):
    count = 0
    for char in text:
        if char in characters_list:
            count += 1
    return count / max(1, len(text)) * 100

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data_dir')
    parser.add_argument('--tmx-dir', default='data_dir/tmx')
    args = parser.parse_args()

    random.seed(961)
    PUNCS += '،؛؟`’‘”“'
    DIACS = ''.join(['َ', 'ً', 'ُ', 'ٌ', 'ِ', 'ٍ', 'ّ', 'ْ'])
    AR_LETTERS += 'ى'

    files = list()
    for (dirpath, dirnames, filenames) in walk(args.tmx_dir):
        for filename in filenames:
            files.append(join(args.tmx_dir, filename))
    
    data = list()
    for file in files:
        print('Processing %s corpus' % file.split(sep)[-1].split('.')[0])
        tree = ET.parse(file)
        root = tree.getroot()

        for elem in tqdm(root[1]):
            ar_text = elem[0][0].text.lower()
            en_text = elem[1][0].text.lower()

            if len(ar_text.split()) > 50 or len(en_text.split()) > 50:
                continue

            if is_url_exist(ar_text) or is_url_exist(en_text):
                continue

            ar_text = remove_characters(ar_text, PUNCS + DIACS)
            en_text = remove_characters(en_text, PUNCS)

            if 100 - existance_percentage(ar_text, AR_LETTERS) > 15 or \
                 100 - existance_percentage(en_text, EN_LETTERS) > 15:
                continue

            data.append((ar_text, en_text))
    random.shuffle(data)

    print('Number of parallel sentences %s' % len(data))

    ar_data, en_data = zip(*data)

    print('Write Arabic text')
    with open(join(args.data_dir, 'ar'), 'w') as ar_file:
        for ar_text in tqdm(ar_data):
            ar_file.write(ar_text + '\n')

    print('Write English text')
    with open(join(args.data_dir, 'en'), 'w') as en_file:
        for en_text in tqdm(en_data):
            en_file.write(en_text + '\n')
