from pathlib import Path
import argparse
import re

from tqdm import tqdm

from bwgpt2.database import ChatLog, ChatRecord


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', '-i', type=str, required=True)
    parser.add_argument('--output-path', '-o', type=str, required=True)
    args = parser.parse_args()

    path = Path(args.input_path)
    records = []

    name_map = dict(y='Yoosung', s='Saeran', z='ZEN', j='Jaehee', v='V', h='Jumin')

    select_patt = re.compile('\s+Select.+?;')
    extract_patt = re.compile(r'^.+?:(.+?);(.+)$')
    for file_path in tqdm(list(path.glob('*/*.txt'))):
        with open(file_path) as f:
            for line in f:
                if 'SelectEnd' in line or ':[' in line or 'Notification' in line:
                    continue
                line = select_patt.sub('{0};', line)
                m = extract_patt.match(line)
                if not m:
                    continue
                name, message = m.group(1), m.group(2)
                if len(name) == 1:
                    print(name)
                    name = name_map.get(name, name)
                message = message.strip()
                records.append(ChatRecord(0, '', name, message))
    ChatLog(records).write_tsv(args.output_path)


if __name__ == '__main__':
    main()
