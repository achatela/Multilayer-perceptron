#include "../includes/Model.hpp"

Model::Model(std::vector<std::vector<float>> inputs, std::vector<std::string> columnNames, int hiddenLayersNumber = 2, int epochs = 100) : _inputLayer(inputs), _columnNames(columnNames), _epochs(epochs)
{
    this->_hiddenLayers.push_back(Layer(this->_inputLayer));
    for (int i = 0; i < hiddenLayersNumber; i++)
    {
        this->_hiddenLayers.push_back(Layer(64, this->_inputLayer.size(), this->_columnNames.size()));
    }

    for (int i = 0; i < 1; i++)
    {
        for (int j = 1; j < this->_hiddenLayers.size(); j++)
        {
            std::cout << j << std::endl;
            this->_hiddenLayers[j].forwardPropagation(this->_hiddenLayers[j - 1]);
        }
    }
}

Model::~Model()
{
}