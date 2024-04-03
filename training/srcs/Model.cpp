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

        std::cout << "epoch " << i + 1 << "/" << epochs << " - loss:";
        this->_hiddenLayers[0].getValidationLoss(validationSet, this->_hiddenLayers, this->_validationLoss, this->_validationAccuracy);
        std::cout << " - val_loss:";
        this->_hiddenLayers[0].getValidationLoss(inputs, this->_hiddenLayers, this->_trainingLoss, this->_trainingAccuracy);
        std::cout << std::endl;
    }
    for (size_t i = 1; i < this->_hiddenLayers.size(); i++)
    {
        this->_modelArchitecture.push_back(std::vector<std::vector<double>>());
        for (auto &neuron : this->_hiddenLayers[i].getNeurons())
        {
            this->_modelArchitecture.back().push_back(std::vector<double>());
            for (auto &weight : neuron.getWeights())
                this->_modelArchitecture.back().back().push_back(weight);
        }
    }

    // print model architecture
    for (size_t i = 0; i < this->_modelArchitecture.size(); i++)
    {
        std::cout << "Layer " << i << std::endl;
        for (size_t j = 0; j < this->_modelArchitecture[i].size(); j++)
        {
            std::cout << "Neuron " << j << std::endl;
            for (size_t k = 0; k < this->_modelArchitecture[i][j].size(); k++)
                std::cout << this->_modelArchitecture[i][j][k] << " ";
            std::cout << std::endl;
        }
    }

    displayGraphs();
}

Model::~Model() {}

void Model::displayGraphs()
{
    std::string validationLossString = "";
    std::string validationAccuracyString = "";
    std::string trainingLossString = "";
    std::string trainingAccuracyString = "";

    for (size_t i = 0; i < this->_validationLoss.size(); i++)
    {
        validationLossString += std::to_string(this->_validationLoss[i]) + " ";
        validationAccuracyString += std::to_string(this->_validationAccuracy[i]) + " ";
        trainingLossString += std::to_string(this->_trainingLoss[i]) + " ";
        trainingAccuracyString += std::to_string(this->_trainingAccuracy[i]) + " ";
    }

    std::string command = "python3 display_graphs.py \"" + validationLossString + "\" \"" + validationAccuracyString + "\" \"" + trainingLossString + "\" \"" + trainingAccuracyString + "\"";
    system(command.c_str());
}