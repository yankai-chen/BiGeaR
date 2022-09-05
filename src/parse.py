"""
@author:chenyankai
@file:parse.py
@time:2021/11/10
"""
import argparse

def parse_args():
    parse = argparse.ArgumentParser(description='Lossless')
    parse.add_argument('--dataset', type=str, default='book',
                       help='accessible datasets from [music, gowalla, yelp2018, book]')
    parse.add_argument('--topks', nargs='+', type=int, default=[20, 40, 60, 80, 100], help='top@k test list')
    parse.add_argument('--train_file', type=str, default='train.txt')
    parse.add_argument('--test_file', type=str, default='test.txt')
    parse.add_argument('--train_batch', type=int, default=2048, help='batch size in training')
    parse.add_argument('--test_batch', type=int, default=100, help='batch size in testing')
    parse.add_argument('--layers', type=int, default=2, help='the layer number')
    parse.add_argument('--dim', type=int, default=256, help='embedding dimension')
    parse.add_argument('--eps', type=float, default=1e-20, help='epsilon in gumbel sampling')
    parse.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parse.add_argument('--weight', type=float, default=1e-4, help='the weight of l2 norm')
    parse.add_argument('--tensorboard', type=bool, default=False, help='enable tensorboard')
    parse.add_argument('--epoch', type=int, default=20)
    parse.add_argument('--seed', type=int, default=2021, help='random seed')
    parse.add_argument('--model', type=str, default='bgr', help='models to be trained from [lossless]')
    parse.add_argument('--neg_ratio', type=int, default=1, help='the ratio of negative sampling')
    parse.add_argument('--save_embed', type=int, default=0, help='save embedding or not')
    parse.add_argument('--lmd2', type=float, default=0.1, help='lambda of for weighting the ranking distillation')
    parse.add_argument('--R', type=int, default=100, help='Top-R of ranking distillation.')
    parse.add_argument('--compute_rank', type=int, default=1)
    parse.add_argument('--norm_a', type=float, default=1., help='normal distribution')

    return parse.parse_args()