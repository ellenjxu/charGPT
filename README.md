# nanoGPT

Modified [nanoGPT](https://github.com/karpathy/nanoGPT) for enwik8 character level transformer.

Implements local-global hybrid self attention.

## quick start

Train a character-level GPT on 100M character of compressed Wikipedia. First, we download it as a single file and turn it from raw text into one large stream of integers:

```
$ python data/enwik8/prepare.py
```

This creates `train.bin`, `val.bin`, `test.bin` with a 90M, 5M, 5M split in that data directory. Now we train a baby GPT with the settings in the [config/train_enwik8_small.py](config/train_enwik8_small.py) config file for baseline model, or [config/train_enwik8_hybrid.py](config/train_enwik8_hybrid.py) for the hybrid SA model:

```
$ python train.py config/train_enwik8_hybrid.py
```

We can sample from the final model checkpoint and see the generated samples in the out_dir, saved as sample.txt files:

```
$ python sample.py --out_dir=out-enwik8-char-hybrid
```

This generates a few samples, for example:

```
A band named &quot;Esperanto masculini&quot; (see also [[Masculine wars]]) shares the [[extreme perfection of language]], i.e. the [[synonym]] of &quot;Esperanto masculini&quot; (see ''[[Esperanto masculini]]''.)  Both sides are used in different ways such as [[Elementary masculini]] (see [[Esperanto masculini]]) and [[Esperanto masculini]] (see [[List of esperanto masculini]]). In the United States, Esperanto masculini is also used in masculini large parts of [[Africa]].
```

## baselines

```
$ python test.py --out_dir=out-enwik8-char-small
$ python test.py --out_dir=out-enwik8-char-hybrid
```

and observe the following losses on test:

| model            | params | test loss |
| ---------------- | ------ | --------- |
| small (baseline) | 12.95M |           |
| hybrid           | 12.95M |           |
