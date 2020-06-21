from collections import defaultdict, deque
from typing import List
import argparse
import getpass
import json
import os
import re
import sys

from tqdm import tqdm
import mysql.connector
import pandas as pd

from bwgpt2.workspace import UUID_RESOLVE_FILENAME, CHAT_LOG_FILENAME
from bwgpt2.database import ChatRecord, ChatLog


def sessionize_dialogue(records: List[ChatRecord], session_max_time_secs=300):
    msg_patt = re.compile(r'^/(msg|message|whisper|w|tell)\s+(.+?)\s+(.+?)$', re.IGNORECASE)
    reply_patt = re.compile(r'^/(r|reply)\s+(.+?)$', re.IGNORECASE)
    sessions = defaultdict(list)
    unresolved_sessions = defaultdict(list)
    last_message_map = dict()
    usernames = deque()

    for record in records:
        if len(usernames) > 500:
            usernames.popleft()
        usernames.append(record.username)
        msg_m = msg_patt.match(record.content)
        reply_m = reply_patt.match(record.content)
        if msg_m is not None:
            target = msg_m.group(2)
            message = msg_m.group(3)
            record.content = message
            found = False
            for username in usernames:
                if username.lower().startswith(target.lower()):
                    key = tuple(sorted([username, record.username]))
                    sessions[key].append(record)
                    found = True
                    break
            if not found:
                key = tuple(sorted([target, record.username]))
                unresolved_sessions[(record.username, target)].append(message)
        elif reply_m is not None:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--username', '-u', type=str, required=True, help='Username')
    parser.add_argument('--password', '-p', action='store_true')
    parser.add_argument('--extract-type', '-xt', type=str, default='all', choices=('all', 'dialogue'))
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

    if args.extract_type == 'all':
        c.execute(f'SELECT time, uuid, username, content FROM {args.table} WHERE type=1')
        records = [ChatRecord(*row) for row in tqdm(c, desc='Fetching records')]
        print('Saving records...', file=sys.stderr)
    elif args.extract_type == 'dialogue':
        c.execute(f'SELECT time, uuid, username, content FROM {args.table} WHERE type=2 and (content like "/r %" or content like "/msg %"' + \
                  'or content like "/w %" or content like "/whisper %" or content like "/tell %" or content like "/reply %" or content like "/message %");')
        records = [ChatRecord(*row) for row in tqdm(c, desc='Fetching records')]
        records = sessionize_dialogue(records)
    ChatLog(records).write_tsv(os.path.join(args.workspace, CHAT_LOG_FILENAME))


if __name__ == '__main__':
    main()
