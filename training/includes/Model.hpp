#include <vector>
#include <string>
#include "Layer.hpp"
#include <iostream>

class Model
{

public:
    Model(std::vector<std::vector<float>> inputs, std::vector<std::string> columnNames, int hiddenLayersNumber, int epochs);
    ~Model();

    std::vector<std::vector<float>> &getInputLayer() { return this->_inputLayer; };
    std::vector<Layer> &getHiddenLayers() { return this->_hiddenLayers; };
    Layer &getOutputLayer() { return this->_outputLayer; };

    void setClassesInputs(std::vector<std::vector<float>> classesInputs);

private:
    std::vector<std::vector<float>> _inputLayer;
    std::vector<Layer> _hiddenLayers;
    Layer _outputLayer;

    std::vector<std::string> _columnNames;

    int _epochs;

    std::vector<std::vector<std::vector<float>>> _classesInputs; // classesInputs[x] every row belonging to class x
};