import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--dataDir", type=str, default='./data/bi', help="the path to datasets")
parser.add_argument("--func", type=str, choices=['variance', 'section'], help="the function of this script")
args = parser.parse_args()

algos = ['noWS', 'cuMBE'] if args.func == 'variance' else ['noRS', 'noES', 'noWS', 'cuMBE']
datas = [
    'DBLP-author'             ,
    'DBpedia_locations'       ,
    'Marvel'                  ,
    'YouTube'                 ,
    'IMDB-actor'              ,
    'stackoverflow'           ,
    'BookCrossing'            ,
    'corporate-leadership'    ,
    'movielens-t-i'           ,
    'movielens-u-i'           ,
    'movielens-u-t'           ,
    'UCforum'                 ,
    'Unicode'                 ,
]

if args.func == None:
    for data in datas:
        print('\33[33m------ {} ------\33[0m'.format(data))
        for algo in algos:
            os.system('./bin/mbe {}/{}.bi {}'.format(args.dataDir, data, algo))
elif args.func == 'variance':
    for data in datas:
        print('\33[33m------ {} ------\33[0m'.format(data))
        for algo in algos:
            os.system('./bin/mbe_devariance {}/{}.bi {} >> result/{}_{}_variance'.format(args.dataDir, data, algo, data, algo))
elif args.func == 'section':
    for data in datas:
        print('\33[33m------ {} ------\33[0m'.format(data))
        for algo in algos:
            os.system('./bin/mbe_desection {}/{}.bi {} >> result/{}_{}_section'.format(args.dataDir, data, algo, data, algo))

if args.func == None:
    pass
elif args.func == 'variance':
    os.system('python utils/plot_variance.py')
elif args.func == 'section':
    os.system('python utils/plot_section.py')
