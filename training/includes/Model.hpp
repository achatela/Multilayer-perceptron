#include "Layer.hpp"

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

class Model
{

public:
    Model(std::vector<std::vector<double>> &inputs, std::vector<std::string> &columnNames, std::vector<std::vector<double>> &validationSet, int epochs, double learningRate, std::vector<double> &hiddenLayersPattern);
    Model(std::string modelWeights);
    ~Model();

    void saveModel();
    void displayGraphs();

private:
    std::vector<std::vector<double>> _inputLayer;
    std::vector<Layer> _hiddenLayers;

    std::vector<std::string> _columnNames;

    std::vector<double> _validationLoss;
    std::vector<double> _validationAccuracy;
    std::vector<double> _trainingLoss;
    std::vector<double> _trainingAccuracy;

    std::vector<std::vector<std::vector<double>>> _modelArchitecture;
};