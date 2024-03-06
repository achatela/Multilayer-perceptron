#include <vector>
#include <string>
#include "Layer.hpp"
#include <iostream>

class Model
{

public:
    Model(std::vector<std::vector<double>> inputs, std::vector<std::string> columnNames, std::vector<std::vector<double>> validationSet, int hiddenLayersNumber, int epochs, double learningRate);
    ~Model();

    std::vector<std::vector<double>> &getInputLayer() { return this->_inputLayer; };
    std::vector<Layer> &getHiddenLayers() { return this->_hiddenLayers; };
    Layer &getOutputLayer() { return this->_outputLayer; };

    void setClassesInputs(std::vector<std::vector<double>> classesInputs);
    std::vector<std::vector<std::vector<double>>> getClassesInputs() { return this->_classesInputs; };

    void setFinalWeights(std::vector<std::vector<double>> finalWeights) { this->finalWeights = finalWeights; };
    std::vector<std::vector<double>> getFinalWeights() { return this->finalWeights; };

    int predictClass(std::vector<double> inputs);

private:
    std::vector<std::vector<double>> _inputLayer;
    std::vector<Layer> _hiddenLayers;
    Layer _outputLayer;

    std::vector<std::string> _columnNames;

    int _epochs;

    std::vector<std::vector<double>> finalWeights;

    std::vector<std::vector<std::vector<double>>> _classesInputs; // classesInputs[x] every row belonging to class x
};