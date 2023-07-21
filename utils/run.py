import os

datas = [
    # 'arXiv_cond-mat'          ,
    # 'bn-mouse_visual-cortex_1',
    'DBLP-author'             ,
    'DBpedia_locations'       ,
    # 'edit-cswikisource'       ,
    # 'edit-guwikibooks'        ,
    # 'edit-hawiktionary'       ,
    # 'gen-20-complete'         ,
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
    # 'wang-tripadvisor'        ,
    # 'Teams'                   ,
    # 'edit-hewikisource'       ,
    # 'edit-enwiktionary'       ,
    # 'edit-enwikisource'       ,
    # 'github'                  ,
    # 'amazon-ratings'          ,
]

# os.system('make')

for data in datas:
    print('\33[33m------ {} ------\33[0m'.format(data))
    os.system('./bin/mbe ~/sorryTT/data/bi/{}.bi 999'.format(data))
    os.system('./../cohesive_subgraph_bipartite/mbbp ~/sorryTT/data/edge/{}.e'.format(data))
    # os.system('./../ParMBE/test/parmbe_test/parmbe_main ~/sorryTT/data/txt/{}.txt'.format(data))
    # os.system('./bin/mbe_nows ~/sorryTT/data/bi/{}.bi 999'.format(data))