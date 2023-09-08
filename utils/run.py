import argparse
import os

algorithms = ['noRS', 'noES', 'noWS', 'cuMBE']
algorithms_variance = ['noWS', 'cuMBE']
datasets = [
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

parser = argparse.ArgumentParser()
parser.add_argument("--func"   , type=str, choices=['variance', 'section'], help="choose the function of this script")
parser.add_argument("--algo"   , type=str, choices=algorithms, nargs='+', help="specify MBE algorithms")
parser.add_argument("--data"   , type=str, choices=datasets, nargs='+', help="specify MBE datasets")
parser.add_argument("--dataDir", type=str, default='./data/bi', help="specify the path to datasets")
parser.add_argument("--no_run" , action='store_true', default=False, help="skip MBE execution")
parser.add_argument("--no_plot", action='store_true', default=False, help="skip plotting")
args = parser.parse_args()

algos = args.algo if args.algo else algorithms if args.func != 'variance' else algorithms_variance
datas = args.data if args.data else datasets

print('algorithms:', algos)
print('datasets:', datas)
print('dataDir:', args.dataDir)

if args.no_run:
    pass
elif args.func == None:
    for data in datas:
        # print('\33[33m====== {} ======\33[0m'.format(data))
        for algo in algos:
            print('------ {} ------ {} ------'.format(data, algo))
            os.system('./bin/mbe {}/{}.bi {}'.format(args.dataDir, data, algo))
elif args.func == 'variance':
    for data in datas:
        # print('\33[33m====== {} ======\33[0m'.format(data))
        for algo in algos:
            print('------ {} ------ {} ------'.format(data, algo))
            os.system('./bin/mbe_devariance {}/{}.bi {} >> result/{}_{}_variance'.format(args.dataDir, data, algo, data, algo))
elif args.func == 'section':
    for data in datas:
        # print('\33[33m====== {} ======\33[0m'.format(data))
        for algo in algos:
            print('------ {} ------ {} ------'.format(data, algo))
            os.system('./bin/mbe_desection {}/{}.bi {} >> result/{}_{}_section'.format(args.dataDir, data, algo, data, algo))

if args.no_plot or args.func == None:
    pass
elif args.func == 'variance':
    os.system('python utils/plot_variance.py')
elif args.func == 'section':
    os.system('python utils/plot_section.py')