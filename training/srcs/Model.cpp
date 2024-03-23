#include "../includes/Model.hpp"

Model::Model(std::vector<std::vector<double>> &inputs, std::vector<std::string> &columnNames, std::vector<std::vector<double>> &validationSet, int epochs, double learningRate, std::vector<double> &hiddenLayersPattern) : _inputLayer(inputs), _columnNames(columnNames)
{
    this->_hiddenLayers.push_back(Layer(this->_inputLayer));
    for (size_t i = 0; i < hiddenLayersPattern.size(); i++)
        this->_hiddenLayers.push_back(Layer(hiddenLayersPattern[i], this->_hiddenLayers.back().getNeurons().size()));
    this->_hiddenLayers.push_back(Layer(2, this->_hiddenLayers.back().getNeurons().size())); // output layer

    for (int i = 0; i < epochs; i++)
    {
        for (std::vector<double> &input : inputs)
        {
            this->_hiddenLayers[1].firstHiddenLayerFeed(input);
            for (size_t j = 2; j < this->_hiddenLayers.size() - 1; j++)
                this->_hiddenLayers[j].hiddenLayerFeed(this->_hiddenLayers[j - 1]);
            this->_hiddenLayers.back().outputLayerFeed(this->_hiddenLayers[this->_hiddenLayers.size() - 2], input);

            this->_hiddenLayers.back().backPropagation(this->_hiddenLayers, input, learningRate);
        }

        double validation_loss = this->_hiddenLayers[0].getValidationLoss(validationSet, this->_hiddenLayers);
        double loss = this->_hiddenLayers[0].getValidationLoss(inputs, this->_hiddenLayers);

        std::cout << "epoch " << i + 1 << "/" << epochs << " - loss: " << loss << " - val_loss: " << validation_loss << std::endl;
    }
}

Model::~Model() {}