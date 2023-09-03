all:
	mkdir -p log
	mkdir -p bin
	mkdir -p result
clean:
	rm log/    -r 2> /dev/null
	rm bin/    -r 2> /dev/null
	rm result/ -r 2> /dev/null