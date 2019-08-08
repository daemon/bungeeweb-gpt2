import argparse
import json
import os
import sys

from pytorch_pretrained_bert import GPT2Tokenizer
from tqdm import tqdm

from bwgpt2.workspace import UUID_RESOLVE_FILENAME, CHAT_LOG_FILENAME
from bwgpt2.database import ChatRecord, ChatLog


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=str, default='output')
    parser.add_argument('--lm-type', type=str, default='global', choices=('dialogue', 'global'))
    parser.add_argument('--max-tokens', default=128, type=int)
    args = parser.parse_args()

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    print('Loading chat log...', file=sys.stderr)
    chat_log = ChatLog.from_tsv(os.path.join(args.workspace, CHAT_LOG_FILENAME))
    if args.lm_type == 'global':
        tokens = []
        for record in tqdm(chat_log.records):
            username = record.username
            if tokens: username = ' ' + username
            tokens.extend(tokenizer.encode(' '.join((username, record.content, '|'))))
            if len(tokens) >= args.max_tokens:
                print(tokenizer.decode(tokens[:args.max_tokens]))
                tokens = []


if __name__ == '__main__':
    main()
