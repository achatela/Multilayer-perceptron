#include "../includes/Model.hpp"

Model::Model(std::vector<std::vector<float>> inputs, std::vector<std::string> columnNames, int hiddenLayersNumber = 2, int epochs = 100) : _inputLayer(inputs), _columnNames(columnNames), _epochs(epochs)
{
    for (int i = 0; i < hiddenLayersNumber; i++)
    {
        this->_hiddenLayers.push_back(Layer(64, this->_inputLayer.size(), this->_columnNames.size()));
    }

    for (int i = 0; i < epochs; i++)
    {
        for (int j = 0; j < this->_hiddenLayers.size(); j++)
        {
            if (j == 0)
                this->_hiddenLayers[j].forwardPropagation(Layer(this->_inputLayer));
            // else
            //     this->_hiddenLayers[j].forwardPropagation(this->_hiddenLayers[j - 1]);
        }
    }
}

Model::~Model()
{
}