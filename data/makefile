all:
	wget http://konect.cc/files/download.tsv.dblp-author.tar.bz2
	tar xvf download.tsv.dblp-author.tar.bz2
	rm download.tsv.dblp-author.tar.bz2
	wget http://konect.cc/files/download.tsv.dbpedia-location.tar.bz2
	tar xvf download.tsv.dbpedia-location.tar.bz2
	rm download.tsv.dbpedia-location.tar.bz2
	wget http://konect.cc/files/download.tsv.marvel.tar.bz2
	tar xvf download.tsv.marvel.tar.bz2
	rm download.tsv.marvel.tar.bz2
	wget http://konect.cc/files/download.tsv.youtube-groupmemberships.tar.bz2
	tar xvf download.tsv.youtube-groupmemberships.tar.bz2
	rm download.tsv.youtube-groupmemberships.tar.bz2
	wget http://konect.cc/files/download.tsv.actor2.tar.bz2
	tar xvf download.tsv.actor2.tar.bz2
	rm download.tsv.actor2.tar.bz2
	wget http://konect.cc/files/download.tsv.stackexchange-stackoverflow.tar.bz2
	tar xvf download.tsv.stackexchange-stackoverflow.tar.bz2
	rm download.tsv.stackexchange-stackoverflow.tar.bz2
	wget http://konect.cc/files/download.tsv.bookcrossing_full-rating.tar.bz2
	tar xvf download.tsv.bookcrossing_full-rating.tar.bz2
	rm download.tsv.bookcrossing_full-rating.tar.bz2
	wget http://konect.cc/files/download.tsv.brunson_corporate-leadership.tar.bz2
	tar xvf download.tsv.brunson_corporate-leadership.tar.bz2
	rm download.tsv.brunson_corporate-leadership.tar.bz2
	wget http://konect.cc/files/download.tsv.movielens-10m_ti.tar.bz2
	tar xvf download.tsv.movielens-10m_ti.tar.bz2
	rm download.tsv.movielens-10m_ti.tar.bz2
	wget http://konect.cc/files/download.tsv.movielens-10m_ui.tar.bz2
	tar xvf download.tsv.movielens-10m_ui.tar.bz2
	rm download.tsv.movielens-10m_ui.tar.bz2
	wget http://konect.cc/files/download.tsv.movielens-10m_ut.tar.bz2
	tar xvf download.tsv.movielens-10m_ut.tar.bz2
	rm download.tsv.movielens-10m_ut.tar.bz2
	wget http://konect.cc/files/download.tsv.opsahl-ucforum.tar.bz2
	tar xvf download.tsv.opsahl-ucforum.tar.bz2
	rm download.tsv.opsahl-ucforum.tar.bz2
	wget http://konect.cc/files/download.tsv.unicodelang.tar.bz2
	tar xvf download.tsv.unicodelang.tar.bz2
	rm download.tsv.unicodelang.tar.bz2
	mkdir -p bi
	g++ gen_bi.cpp -o gen_bi
	./gen_bi ./dblp-author/out.dblp-author ./bi/DBLP-author.bi 7 0
	./gen_bi ./dbpedia-location/out.dbpedia-location ./bi/DBpedia_locations.bi 7 0
	./gen_bi ./marvel/out.marvel ./bi/Marvel.bi 3 0
	./gen_bi ./youtube-groupmemberships/out.youtube-groupmemberships ./bi/YouTube.bi 3 0
	./gen_bi ./actor2/out.actor2 ./bi/IMDB-actor.bi 3 0
	./gen_bi ./stackexchange-stackoverflow/out.stackexchange-stackoverflow ./bi/stackoverflow.bi 7 2
	./gen_bi ./bookcrossing_full-rating/out.bookcrossing_full-rating_full-rating ./bi/BookCrossing.bi 7 0
	./gen_bi ./brunson_corporate-leadership/out.brunson_corporate-leadership_corporate-leadership ./bi/corporate-leadership.bi 7 0
	./gen_bi ./movielens-10m_ti/out.movielens-10m_ti ./bi/movielens-t-i.bi 3 2
	./gen_bi ./movielens-10m_ui/out.movielens-10m_ui ./bi/movielens-u-i.bi 3 2
	./gen_bi ./movielens-10m_ut/out.movielens-10m_ut ./bi/movielens-u-t.bi 3 2
	./gen_bi ./opsahl-ucforum/out.opsahl-ucforum ./bi/UCforum.bi 7 2
	./gen_bi ./unicodelang/out.unicodelang ./bi/Unicode.bi 3 1
dataset:
	wget http://konect.cc/files/download.tsv.dblp-author.tar.bz2
	tar xvf download.tsv.dblp-author.tar.bz2
	rm download.tsv.dblp-author.tar.bz2
	wget http://konect.cc/files/download.tsv.dbpedia-location.tar.bz2
	tar xvf download.tsv.dbpedia-location.tar.bz2
	rm download.tsv.dbpedia-location.tar.bz2
	wget http://konect.cc/files/download.tsv.marvel.tar.bz2
	tar xvf download.tsv.marvel.tar.bz2
	rm download.tsv.marvel.tar.bz2
	wget http://konect.cc/files/download.tsv.youtube-groupmemberships.tar.bz2
	tar xvf download.tsv.youtube-groupmemberships.tar.bz2
	rm download.tsv.youtube-groupmemberships.tar.bz2
	wget http://konect.cc/files/download.tsv.actor2.tar.bz2
	tar xvf download.tsv.actor2.tar.bz2
	rm download.tsv.actor2.tar.bz2
	wget http://konect.cc/files/download.tsv.stackexchange-stackoverflow.tar.bz2
	tar xvf download.tsv.stackexchange-stackoverflow.tar.bz2
	rm download.tsv.stackexchange-stackoverflow.tar.bz2
	wget http://konect.cc/files/download.tsv.bookcrossing_full-rating.tar.bz2
	tar xvf download.tsv.bookcrossing_full-rating.tar.bz2
	rm download.tsv.bookcrossing_full-rating.tar.bz2
	wget http://konect.cc/files/download.tsv.brunson_corporate-leadership.tar.bz2
	tar xvf download.tsv.brunson_corporate-leadership.tar.bz2
	rm download.tsv.brunson_corporate-leadership.tar.bz2
	wget http://konect.cc/files/download.tsv.movielens-10m_ti.tar.bz2
	tar xvf download.tsv.movielens-10m_ti.tar.bz2
	rm download.tsv.movielens-10m_ti.tar.bz2
	wget http://konect.cc/files/download.tsv.movielens-10m_ui.tar.bz2
	tar xvf download.tsv.movielens-10m_ui.tar.bz2
	rm download.tsv.movielens-10m_ui.tar.bz2
	wget http://konect.cc/files/download.tsv.movielens-10m_ut.tar.bz2
	tar xvf download.tsv.movielens-10m_ut.tar.bz2
	rm download.tsv.movielens-10m_ut.tar.bz2
	wget http://konect.cc/files/download.tsv.opsahl-ucforum.tar.bz2
	tar xvf download.tsv.opsahl-ucforum.tar.bz2
	rm download.tsv.opsahl-ucforum.tar.bz2
	wget http://konect.cc/files/download.tsv.unicodelang.tar.bz2
	tar xvf download.tsv.unicodelang.tar.bz2
	rm download.tsv.unicodelang.tar.bz2
bipartite:
	mkdir -p bi
	g++ gen_bi.cpp -o gen_bi
	./gen_bi ./dblp-author/out.dblp-author ./bi/DBLP-author.bi 7 0
	./gen_bi ./dbpedia-location/out.dbpedia-location ./bi/DBpedia_locations.bi 7 0
	./gen_bi ./marvel/out.marvel ./bi/Marvel.bi 3 0
	./gen_bi ./youtube-groupmemberships/out.youtube-groupmemberships ./bi/YouTube.bi 3 0
	./gen_bi ./actor2/out.actor2 ./bi/IMDB-actor.bi 3 0
	./gen_bi ./stackexchange-stackoverflow/out.stackexchange-stackoverflow ./bi/stackoverflow.bi 7 2
	./gen_bi ./bookcrossing_full-rating/out.bookcrossing_full-rating_full-rating ./bi/BookCrossing.bi 7 0
	./gen_bi ./brunson_corporate-leadership/out.brunson_corporate-leadership_corporate-leadership ./bi/corporate-leadership.bi 7 0
	./gen_bi ./movielens-10m_ti/out.movielens-10m_ti ./bi/movielens-t-i.bi 3 2
	./gen_bi ./movielens-10m_ui/out.movielens-10m_ui ./bi/movielens-u-i.bi 3 2
	./gen_bi ./movielens-10m_ut/out.movielens-10m_ut ./bi/movielens-u-t.bi 3 2
	./gen_bi ./opsahl-ucforum/out.opsahl-ucforum ./bi/UCforum.bi 7 2
	./gen_bi ./unicodelang/out.unicodelang ./bi/Unicode.bi 3 1
clean:
	rm ./gen_bi                      -r 2> /dev/null
	rm bi/                           -r 2> /dev/null
	rm dblp-author/                  -r 2> /dev/null
	rm dbpedia-location/             -r 2> /dev/null
	rm marvel/                       -r 2> /dev/null
	rm youtube-groupmemberships/     -r 2> /dev/null
	rm actor2/                       -r 2> /dev/null
	rm stackexchange-stackoverflow/  -r 2> /dev/null
	rm bookcrossing_full-rating/     -r 2> /dev/null
	rm brunson_corporate-leadership/ -r 2> /dev/null
	rm movielens-10m_ti/             -r 2> /dev/null
	rm movielens-10m_ui/             -r 2> /dev/null
	rm movielens-10m_ut/             -r 2> /dev/null
	rm opsahl-ucforum/               -r 2> /dev/null
	rm unicodelang/                  -r 2> /dev/null