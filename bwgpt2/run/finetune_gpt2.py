from collections import Counter
import argparse
import functools
import math
import random
import time

from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from torch.distributions.categorical import Categorical
import torch
import torch.nn as nn
import torch.utils.data as tud

from .args import add_dict_options, opt, OptionEnum
from bwgpt2.utils import set_seed, dual_print
from bwgpt2.data import tokenize_batch, FlatFileDataset


EOS_TOKEN = '<|endoftext|>'


ARGS = [
    OptionEnum.LEARNING_RATE.value.default(5e-5),
    OptionEnum.TRAIN_BATCH_SIZE.value.default(15),
    OptionEnum.EVAL_BATCH_SIZE.value.default(15),
    OptionEnum.NUM_TRAIN_EPOCHS.value,
    OptionEnum.SEED.value.default(0),
    OptionEnum.WARMUP_PROPORTION.value.default(0.1),
    opt('--data-folder', type=str, required=True),
    opt('--weight-decay', type=float, default=1e-3),
    opt('--log-interval', type=int, default=100),
    opt('--save', type=str, default='gpt2.pt'),
    opt('--resume', type=str),
    opt('--cache-dir', type=str),
    opt('--no-train', action='store_false', dest='do_train'),
    opt('--gpt2-model', type=str, default='gpt2'),
    opt('--test-eval', action='store_true'),
    opt('--use-sos', action='store_true'),
    opt('--no-drop-last', action='store_false', dest='drop_last'),
    opt('--reset', action='store_true')
]


def gpt_encode(tokenizer, examples, eos=True, max_len=128):
    eos_idx = tokenizer.encoder[EOS_TOKEN]
    eos_append = [eos_idx] if eos else []
    tokens_lst = []
    for example in examples:
        tokens = tokenizer.encode(example)
        if eos: tokens[-1] = eos_idx
        tokens_lst.append(tokens)
    tokens_mask = [[1] * len(x) for x in tokens_lst]
    old_max_len = max_len
    max_len = min(max(map(len, tokens_lst)), max_len)
    tokens_lst = [x[:max_len] for x in tokens_lst]
    tokens_mask = [x[:max_len] for x in tokens_mask]
    total_in_chars = 0
    tokens_lst = [x + [0] * (max_len - len(x)) for x in tokens_lst]
    tokens_mask = [x + [0] * (max_len - len(x)) for x in tokens_mask]
    if total_in_chars == 0: total_in_chars = sum(min(len(ex) - 1 + len(eos_append), old_max_len) for ex in examples)
    return tokens_lst, tokens_mask, total_in_chars


def sample_query(model, tokenizer, text, n=128):
    model.cuda()
    new_toks = [tokenizer.decoder[x] for x in tokenizer.encode(text)]
    full_text = ''
    past = None
    hist_len = 0
    new_len = len(new_toks)
    for _ in range(n):
        if new_len + hist_len > 1024:
            return ''
        inp_ = [tokenizer.encoder[x] for x in new_toks]
        examples = torch.LongTensor(inp_).unsqueeze(0).cuda()
        model.eval()
        with torch.no_grad():
            output, present = model(examples, past=past)
            output = output.permute(0, 2, 1)[:, :, -1].contiguous().view(-1)
        try:
            text = tokenizer.decode(examples.view(-1).tolist() + [Categorical(logits=output).sample().item()])
            new_toks = [tokenizer.decoder[x] for x in tokenizer.encode(text)]
            if EOS_TOKEN in text:
                return full_text + text
            found_idx = -1
            for idx, x in enumerate(new_toks):
                if 'Ġ' in x:
                    found_idx = idx
            if found_idx > 0:
                full_text += tokenizer.decode([tokenizer.encoder[x] for x in new_toks[:found_idx]])
                hist_len += found_idx
                new_len = len(new_toks)
                new_toks = new_toks[found_idx:]
                past = present
        except KeyError:
            break
    return full_text + text


def main():
    def evaluate(data_source, split_encode=False):
        model.eval()
        total_loss = 0
        total_words = 0
        total_n = 0
        batch_idx = 0
        for examples in data_source:
            total_words += sum(len(x.split()) for x in examples)
            try:
                examples, mask, total_chars = gpt_encode(tokenizer, examples)
            except KeyError:
                continue
            mask = torch.Tensor(mask).cuda()
            examples = torch.LongTensor(examples).cuda()

            with torch.no_grad():
                output = model(examples[:, :-1])[0].permute(0, 2, 1)
            targets = examples[:, 1:]
            crit = criterion(output, targets)
            mask_tot = mask[:, 1:].sum()
            raw_loss = (crit * mask[:, 1:]).sum() / mask_tot
            loss = raw_loss

            total_loss += raw_loss.item() * mask_tot.item()
            total_n += total_chars
            # print(total_loss / (math.log(2) * total_n))

        cur_loss = total_loss / total_n
        elapsed = time.time() - start_time
        word_ppl = math.exp(total_loss / total_words)
        dual_print('-' * 89)
        dual_print('| end of epoch {:3d} | lr {:05.5f} | ms/batch {:5.2f} | '
                'loss {:5.2f} | ppl {:8.2f} | bpc {:8.3f}'.format(
            epoch, optimizer.param_groups[0]['lr'],
            elapsed * 1000 / args.log_interval, cur_loss, word_ppl, cur_loss / math.log(2)))
        dual_print('-' * 89)
        return cur_loss / math.log(2)

    parser = argparse.ArgumentParser()
    add_dict_options(parser, ARGS)
    args = parser.parse_args()
    set_seed(args.seed)

    tokenizer = GPT2Tokenizer.from_pretrained(args.gpt2_model, cache_dir=args.cache_dir)
    model = GPT2LMHeadModel.from_pretrained(args.gpt2_model, cache_dir=args.cache_dir)
    if args.reset: model.apply(model.init_weights)
    train_ds, dev_ds, test_ds = FlatFileDataset.splits(args.data_folder)
    criterion = nn.CrossEntropyLoss(reduction='none')

    train_loader = tud.DataLoader(train_ds, batch_size=args.train_batch_size, shuffle=True, drop_last=args.drop_last)
    dev_loader = tud.DataLoader(dev_ds, batch_size=args.eval_batch_size, shuffle=False, drop_last=args.drop_last)
    test_loader = tud.DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False, drop_last=args.drop_last)

    no_decay = ['bias']
    params = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in params if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    num_train_optimization_steps = args.num_train_epochs * len(train_loader)
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate)
    t_total = len(train_loader) * args.num_train_epochs
    num_warmup_steps = args.warmup_proportion * t_total
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total)

    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=lambda s, l: s))
    if args.test_eval:
        while True:
            query = input("> ")
            print(sample_query(model, tokenizer, query))
        return

    model = nn.DataParallel(model).cuda()
    start_time = time.time()
    best_bpc = 1000000

    if not args.do_train:
        evaluate(test_loader, split_encode=False)
        return

    for epoch in range(args.num_train_epochs):
        epoch += 1
        total_loss = 0
        total_words = 0
        total_n = 0
        batch_idx = 0
        for examples in train_loader:
            model.train()
            total_words += sum(len(x.split()) for x in examples)
            try:
                examples, mask, total_chars = gpt_encode(tokenizer, examples)
            except KeyError:
                dual_print('Skipped batch')
                continue
            mask = torch.Tensor(mask).cuda()
            examples = torch.LongTensor(examples).cuda()
            optimizer.zero_grad()

            output = model(examples[:, :-1])[0].permute(0, 2, 1)
            targets = examples[:, 1:]
            crit = criterion(output, targets)
            mask_tot = mask[:, 1:].sum()
            raw_loss = (crit * mask[:, 1:]).sum() / mask_tot

            loss = raw_loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += raw_loss.item() * mask_tot.item()
            total_n += total_chars
            if batch_idx % args.log_interval == 0 and batch_idx > 0:
                cur_loss = total_loss / total_n
                word_ppl = math.exp(total_loss / total_words)
                total_words = 0
                elapsed = time.time() - start_time
                dual_print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | '
                        'loss {:5.2f} | ppl {:8.2f} | bpc {:8.3f}'.format(
                    epoch, batch_idx, len(train_loader), optimizer.param_groups[0]['lr'],
                    elapsed * 1000 / args.log_interval, cur_loss, word_ppl, cur_loss / math.log(2)))
                total_loss = 0
                total_n = 0
                start_time = time.time()
            batch_idx += 1
        bpc = evaluate(dev_loader)
        if bpc < best_bpc:
            best_bpc = bpc
            torch.save(model.module.state_dict(), args.save)
    evaluate(test_loader)


if __name__ == '__main__':
    main()