import argparse
import getpass
import json
import os
import sys

from tqdm import tqdm
import mysql.connector
import pandas as pd

from bwgpt2.workspace import UUID_RESOLVE_FILENAME, CHAT_LOG_FILENAME
from bwgpt2.database import ChatRecord, ChatLog


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--username', '-u', type=str, required=True, help='Username')
    parser.add_argument('--password', '-p', action='store_true')
    parser.add_argument('--host', type=str, default='127.0.0.1')
    parser.add_argument('--table', type=str, default='bungeeweb_log')
    parser.add_argument('--workspace', type=str, default='output')
    parser.add_argument('database', type=str)
    args = parser.parse_args()

    password = getpass.getpass() if args.password else ''
    conn = mysql.connector.connect(host=args.host, user=args.username, passwd=password)
    c = conn.cursor()
    c.execute(f'USE {args.database}')
    os.makedirs(args.workspace, exist_ok=True)

    uuid_path = os.path.join(args.workspace, UUID_RESOLVE_FILENAME)
    if os.path.isfile(uuid_path):
        print('Loaded cached UUID map.', file=sys.stderr)
        with open(uuid_path) as f: uuid_map = json.load(f)
    else:
        print('Building UUID map...', file=sys.stderr)
        c.execute(f'SELECT uuid, username FROM {args.table} WHERE type=1 GROUP BY uuid ORDER BY TIME DESC')
        uuid_map = {uuid: username for uuid, username in c}
        with open(uuid_path, 'w') as f:
            json.dump(uuid_map, f)

    c.execute(f'SELECT time, uuid, username, content FROM {args.table} WHERE type=1')
    records = [ChatRecord(*row) for row in tqdm(c, desc='Fetching records')]
    print('Saving records...', file=sys.stderr)
    ChatLog(records).write_tsv(os.path.join(args.workspace, CHAT_LOG_FILENAME))


if __name__ == '__main__':
    main()
