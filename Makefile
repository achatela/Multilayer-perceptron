CXX = c++
CXXFLAGS = -O3 -W -Werror -Wall -Wextra

SRCS_PATH = training/srcs/
SRCS = Model.cpp Layer.cpp Neuron.cpp

INC_PATH = training/includes/
INCLUDES = Model.hpp Layer.hpp Neuron.hpp

FULL_SRCS = $(addprefix $(SRCS_PATH), $(SRCS))
FULL_INC = $(addprefix $(INC_PATH), $(INCLUDES))

all: train predict

shuffle:
	python3 utils/separate_dataset.py data.csv
	./train data_training.csv data_test.csv 1000 0.0023 "8 4"
	./predict model.txt data_training.csv

evaluation:
	python3 utils/evaluation.py data.csv
	./train data_training.csv data_test.csv 1000 0.0023 "8 4"
	./predict model.txt data_training.csv

train: training/training.cpp $(FULL_SRCS) $(FULL_INC)
	$(CXX) $(CXXFLAGS) -o $@ $(filter %.cpp, $^)

predict: training/predict.cpp $(FULL_SRCS) $(FULL_INC)
	$(CXX) $(CXXFLAGS) -o $@ $(filter %.cpp, $^)

re: fclean all

clean:
	rm -f train predict

fclean:
	rm -f train predict model.txt data_test.csv data_training.csv

.PHONY: all clean