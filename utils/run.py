import os

# datas = [
#     'arXiv_cond-ma'           ,
#     'bn-mouse_visual-cortex_1',
#     'BookCrossing'            ,
#     'DBpedia_locations'       ,
#     'edit-cswikisource'       ,
#     'edit-hawiktionary'       ,
#     'IMDB-actor'              ,
#     'Marvel'                  ,
#     'stackoverflow'           ,
#     'YouTube'                 ,
# ]

datas = [
    # 'bn-mouse_visual-cortex_1',
    'DBpedia_locations'       ,
    # 'edit-cswikisource'       ,
    # 'edit-hawiktionary'       ,
    'Marvel'                  ,
    'YouTube'                 ,
    'IMDB-actor'              ,
    # 'stackoverflow'           ,
    # 'BookCrossing'            ,
    # 'wang-tripadvisor'        ,
    # 'Teams'                   ,
    # 'edit-hewikisource'       ,
    # 'edit-enwiktionary'       ,
    # 'edit-enwikisource'       ,
    # 'github'                  ,
    # 'amazon-ratings'          ,
]

os.system('make debug')

for data in datas:
    print('\33[33m------ {} ------\33[0m'.format(data))
    os.system('./bin/mbe ~/sorryTT/data/bi/{}.bi 999'.format(data))