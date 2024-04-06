#include "Layer.hpp"

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

class Model
{

public:
    Model(std::vector<std::vector<double>> &inputs, std::vector<std::vector<double>> &validationSet, int epochs = 1000, double learningRate = 0.001, std::vector<double> hiddenLayersPattern = {8, 4});
    Model(std::string modelWeights, std::vector<std::vector<double>> &predictionSet);
    ~Model();

    std::vector<double> loadModel(std::string modelWeights, std::vector<std::vector<double>> &predictionSet);
    void saveModel();
    void displayGraphs();

private:
    std::vector<std::vector<double>> _inputLayer;
    std::vector<Layer> _hiddenLayers;

    std::vector<double> _validationLoss;
    std::vector<double> _validationAccuracy;
    std::vector<double> _trainingLoss;
    std::vector<double> _trainingAccuracy;

    std::vector<std::vector<std::vector<double>>> _modelArchitecture;
};