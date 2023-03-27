FLAGS_TLP = -O3 --compiler-options -Wall -Xptxas -v
# FLAGS_TLP = --compiler-options -Wall -Xptxas -v
SRC_PATH = ./SpMM
BIN_PATH = ./bin
LOG_PATH = ./log
COMP_LOG = compile
ALGO_MBE = mbe

all:
	@echo "#### Compile Logs ####" > $(LOG_PATH)/$(COMP_LOG).log
	@echo "\n## $(SRC_PATH)/$(ALGO_MBE).cu ##\n" >> $(LOG_PATH)/$(COMP_LOG).log
	nvcc $(FLAGS_TLP) $(SRC_PATH)/$(ALGO_MBE).cu -o $(BIN_PATH)/$(ALGO_MBE) 2>> $(LOG_PATH)/$(COMP_LOG).log
clean:
	rm $(BIN_PATH)/$(ALGO_MBE) -r 2> /dev/null