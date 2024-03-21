#include <vector>
#include <string>
#include "Layer.hpp"
#include <iostream>

class Model
{

public:
    Model(std::vector<std::vector<double>> inputs, std::vector<std::string> columnNames, std::vector<std::vector<double>> validationSet, int hiddenLayersNumber, int epochs, double learningRate);
    ~Model();

private:
    std::vector<std::vector<double>> _inputLayer;
    std::vector<Layer> _hiddenLayers;

    std::vector<std::string> _columnNames;
};