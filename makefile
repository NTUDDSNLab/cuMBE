FLAGS_TLP = --compiler-options -Wall -Xptxas -v
SRC_PATH = ./src
BIN_PATH = ./bin
LOG_PATH = ./log
COMP_LOG = compile
SRC_NAME = main
BIN_NAME = mbe

all:
	@echo "#### Compile Logs ####" > $(LOG_PATH)/$(COMP_LOG).log
	@echo "\n## $(SRC_PATH)/$(ALGO_MBE).cu ##\n" >> $(LOG_PATH)/$(COMP_LOG).log
	nvcc $(FLAGS_TLP) $(SRC_PATH)/$(SRC_NAME).cu -o $(BIN_PATH)/$(BIN_NAME) 2>> $(LOG_PATH)/$(COMP_LOG).log
debug:
	@echo "#### Compile Logs ####" > $(LOG_PATH)/$(COMP_LOG).log
	@echo "\n## $(SRC_PATH)/$(ALGO_MBE).cu ##\n" >> $(LOG_PATH)/$(COMP_LOG).log
	nvcc $(FLAGS_TLP) -DDEBUG $(SRC_PATH)/$(SRC_NAME).cu -o $(BIN_PATH)/$(BIN_NAME) 2>> $(LOG_PATH)/$(COMP_LOG).log
clean:
	rm $(BIN_PATH)/$(ALGO_MBE) -r 2> /dev/null