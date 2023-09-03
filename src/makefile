FLAGS = -O3 --compiler-options -Wall -Xptxas -v
SRC_PATH = .
BIN_PATH = ./../bin
LOG_PATH = ./../log
COMP_LOG = compile
SRC_NAME = main
BIN_NAME = mbe

all:
	@echo "#### Compile Logs ####" > $(LOG_PATH)/$(COMP_LOG).log
	@echo "\n## $(SRC_PATH)/$(ALGO_MBE).cu ##\n" >> $(LOG_PATH)/$(COMP_LOG).log
	nvcc $(FLAGS) $(SRC_PATH)/$(SRC_NAME).cu -o $(BIN_PATH)/$(BIN_NAME) 2>> $(LOG_PATH)/$(COMP_LOG).log
devariance:
	@echo "#### Compile Logs ####" > $(LOG_PATH)/$(COMP_LOG)_devariance.log
	@echo "\n## $(SRC_PATH)/$(ALGO_MBE).cu ##\n" >> $(LOG_PATH)/$(COMP_LOG)_devariance.log
	nvcc $(FLAGS) -DDEVARIANCE $(SRC_PATH)/$(SRC_NAME).cu -o $(BIN_PATH)/$(BIN_NAME)_devariance 2>> $(LOG_PATH)/$(COMP_LOG)_devariance.log
desection:
	@echo "#### Compile Logs ####" > $(LOG_PATH)/$(COMP_LOG)_desection.log
	@echo "\n## $(SRC_PATH)/$(ALGO_MBE).cu ##\n" >> $(LOG_PATH)/$(COMP_LOG)_desection.log
	nvcc $(FLAGS) -DDESECTION $(SRC_PATH)/$(SRC_NAME).cu -o $(BIN_PATH)/$(BIN_NAME)_desection 2>> $(LOG_PATH)/$(COMP_LOG)_desection.log
debug:
	@echo "#### Compile Logs ####" > $(LOG_PATH)/$(COMP_LOG)_debug.log
	@echo "\n## $(SRC_PATH)/$(ALGO_MBE).cu ##\n" >> $(LOG_PATH)/$(COMP_LOG)_debug.log
	nvcc $(FLAGS) -DDEBUG $(SRC_PATH)/$(SRC_NAME).cu -o $(BIN_PATH)/$(BIN_NAME)_debug 2>> $(LOG_PATH)/$(COMP_LOG)_debug.log
allFunc:
	@echo "#### Compile Logs ####" > $(LOG_PATH)/$(COMP_LOG).log
	@echo "\n## $(SRC_PATH)/$(ALGO_MBE).cu ##\n" >> $(LOG_PATH)/$(COMP_LOG).log
	nvcc $(FLAGS) $(SRC_PATH)/$(SRC_NAME).cu -o $(BIN_PATH)/$(BIN_NAME) 2>> $(LOG_PATH)/$(COMP_LOG).log
	@echo "#### Compile Logs ####" > $(LOG_PATH)/$(COMP_LOG)_devariance.log
	@echo "\n## $(SRC_PATH)/$(ALGO_MBE).cu ##\n" >> $(LOG_PATH)/$(COMP_LOG)_devariance.log
	nvcc $(FLAGS) -DDEVARIANCE $(SRC_PATH)/$(SRC_NAME).cu -o $(BIN_PATH)/$(BIN_NAME)_devariance 2>> $(LOG_PATH)/$(COMP_LOG)_devariance.log
	@echo "#### Compile Logs ####" > $(LOG_PATH)/$(COMP_LOG)_desection.log
	@echo "\n## $(SRC_PATH)/$(ALGO_MBE).cu ##\n" >> $(LOG_PATH)/$(COMP_LOG)_desection.log
	nvcc $(FLAGS) -DDESECTION $(SRC_PATH)/$(SRC_NAME).cu -o $(BIN_PATH)/$(BIN_NAME)_desection 2>> $(LOG_PATH)/$(COMP_LOG)_desection.log
	@echo "#### Compile Logs ####" > $(LOG_PATH)/$(COMP_LOG)_debug.log
	@echo "\n## $(SRC_PATH)/$(ALGO_MBE).cu ##\n" >> $(LOG_PATH)/$(COMP_LOG)_debug.log
	nvcc $(FLAGS) -DDEBUG $(SRC_PATH)/$(SRC_NAME).cu -o $(BIN_PATH)/$(BIN_NAME)_debug 2>> $(LOG_PATH)/$(COMP_LOG)_debug.log
clean:
	rm $(BIN_PATH)/$(ALGO_MBE)                -r 2> /dev/null
	rm $(LOG_PATH)/$(COMP_LOG).log            -r 2> /dev/null
	rm $(BIN_PATH)/$(ALGO_MBE)_devariance     -r 2> /dev/null
	rm $(LOG_PATH)/$(COMP_LOG)_devariance.log -r 2> /dev/null
	rm $(BIN_PATH)/$(ALGO_MBE)_desection      -r 2> /dev/null
	rm $(LOG_PATH)/$(COMP_LOG)_desection.log  -r 2> /dev/null
	rm $(BIN_PATH)/$(ALGO_MBE)_debug          -r 2> /dev/null
	rm $(LOG_PATH)/$(COMP_LOG)_debug.log      -r 2> /dev/null
	