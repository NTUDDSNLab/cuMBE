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
    'bn-mouse_visual-cortex_1',
    'DBpedia_locations'       ,
    'edit-cswikisource'       ,
    'edit-hawiktionary'       ,
    'IMDB-actor'              ,
    'Marvel'                  ,
    'YouTube'                 ,
    # 'BookCrossing'            ,
    # 'stackoverflow'           ,
    # 'wang-tripadvisor'        ,
    # 'Teams'                   ,
    # 'edit-hewikisource'       ,
    # 'edit-enwiktionary'       ,
    # 'edit-enwikisource'       ,
    # 'github'                  ,
    # 'amazon-ratings'          ,
]

os.system('make')

for data in datas:
    print('\33[33m------ {} ------\33[0m'.format(data))
    os.system('./bin/mbe ~/sorryTT/data/bi/{}.bi 999'.format(data))