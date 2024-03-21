#include "../includes/Model.hpp"

Model::Model(std::vector<std::vector<double>> inputs, std::vector<std::string> columnNames, std::vector<std::vector<double>> validationSet, int hiddenLayersNumber = 2, int epochs = 100, double learningRate = 0.1) : _inputLayer(inputs), _outputLayer(Layer(2, this->_inputLayer.size(), columnNames.size(), true)), _columnNames(columnNames)
{
    int neuronsNumber = 16; // 16 0.1 best result
    this->_hiddenLayers.push_back(Layer(this->_inputLayer));
    int weightsNumber = columnNames.size();
    for (int i = 0; i < hiddenLayersNumber; i++)
    {
        this->_hiddenLayers.push_back(Layer(neuronsNumber, weightsNumber, weightsNumber, weightsNumber));
        weightsNumber = neuronsNumber;
        neuronsNumber /= 2;
    }
    this->_hiddenLayers.push_back(Layer(2, weightsNumber, weightsNumber, weightsNumber, true)); // TODO change 2 to be the number of classes detected in the dataset

    for (int i = 0; i < epochs; i++)
    {
        double lossSum = 0;
        for (std::vector<double> input : inputs)
        {
            std::vector<double> networkWeights;
            for (size_t j = 1; j < this->_hiddenLayers.size(); j++)
            {
                if (j == this->_hiddenLayers.size() - 1)
                    lossSum += this->_hiddenLayers[j].feedForward(this->_hiddenLayers[j - 1], 2, input);
                else if (j == 1)
                    this->_hiddenLayers[j].feedForward(this->_hiddenLayers[j - 1], 0, input);
                else
                    this->_hiddenLayers[j].feedForward(this->_hiddenLayers[j - 1], 1, input);
            }
            this->_hiddenLayers.back().backPropagation(this->_hiddenLayers, input, learningRate);
        }
        std::vector<double> probabilities;
        for (auto &output : this->_hiddenLayers.back().getNeurons())
            probabilities.push_back(output.getOutput());
        (void)validationSet;
        double validation_loss = this->_hiddenLayers.back().getValidationLoss(validationSet, this->_hiddenLayers);
        double loss = this->_hiddenLayers.back().getValidationLoss(inputs, this->_hiddenLayers); // lossSum / inputs.size();
        std::cout << std::endl
                  << "epoch " << i + 1 << "/" << epochs << " - loss: " << loss << " - val_loss: " << validation_loss << std::endl;
    }
}

Model::~Model()
{
}