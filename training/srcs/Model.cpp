#include "../includes/Model.hpp"

Model::Model(std::vector<std::vector<double>> inputs, std::vector<std::string> columnNames, std::vector<std::vector<double>> validationSet, int hiddenLayersNumber = 2, int epochs = 100, double learningRate = 0.1) : _inputLayer(inputs), _columnNames(columnNames)
{
    int neuronsNumber = 16; // 16 0.1 best result
    int weightsNumber = columnNames.size();

    this->_hiddenLayers.push_back(Layer(this->_inputLayer)); // input layer
    for (int i = 0; i < hiddenLayersNumber; i++)
    {
        this->_hiddenLayers.push_back(Layer(neuronsNumber, weightsNumber)); // hidden layers
        weightsNumber = neuronsNumber;
        neuronsNumber /= 2;
    }
    this->_hiddenLayers.push_back(Layer(2, weightsNumber)); // output layer

    for (int i = 0; i < epochs; i++)
    {
        for (std::vector<double> input : inputs)
        {
            this->_hiddenLayers[1].firstHiddenLayerFeed(input);

            for (size_t j = 2; j < this->_hiddenLayers.size() - 1; j++)
                this->_hiddenLayers[j].hiddenLayerFeed(this->_hiddenLayers[j - 1]);

            this->_hiddenLayers.back().outputLayerFeed(this->_hiddenLayers[this->_hiddenLayers.size() - 2], input);
            this->_hiddenLayers.back().backPropagation(this->_hiddenLayers, input, learningRate);
        }

        double validation_loss = this->_hiddenLayers.back().getValidationLoss(validationSet, this->_hiddenLayers);
        double loss = this->_hiddenLayers.back().getValidationLoss(inputs, this->_hiddenLayers);
        std::cout << std::endl
                  << "epoch " << i + 1 << "/" << epochs << " - loss: " << loss << " - val_loss: " << validation_loss << std::endl;
    }
}

Model::~Model() {}