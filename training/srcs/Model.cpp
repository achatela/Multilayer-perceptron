#include "../includes/Model.hpp"

Model::Model(std::string modelWeights, std::vector<std::vector<double>> &predictionSet)
{
    std::vector<double> biases = loadModel(modelWeights);

    this->_hiddenLayers.push_back(Layer(this->_inputLayer));
    for (size_t i = 0; i < this->_modelArchitecture.size(); i++)
    {
        this->_hiddenLayers.push_back(Layer(this->_modelArchitecture[i], biases[i]));
    }

    std::cout << "loss:";
    this->_hiddenLayers[0].getValidationLoss(predictionSet, this->_hiddenLayers, this->_validationLoss, this->_validationAccuracy);
    std::cout << " - accuracy:" << this->_validationAccuracy.back() << std::endl;
}

Model::Model(std::vector<std::vector<double>> &inputs, std::vector<std::vector<double>> &validationSet, int epochs, double learningRate, std::vector<double> hiddenLayersPattern) : _inputLayer(inputs)
{
    this->_hiddenLayers.push_back(Layer(this->_inputLayer));
    int numberWeights = inputs[0].size();
    for (size_t i = 0; i < hiddenLayersPattern.size(); i++)
    {
        this->_hiddenLayers.push_back(Layer(hiddenLayersPattern[i], numberWeights));
        numberWeights = hiddenLayersPattern[i];
    }
    this->_hiddenLayers.push_back(Layer(2, numberWeights)); // output layer

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
        this->_hiddenLayers[0].getValidationLoss(inputs, this->_hiddenLayers, this->_trainingLoss, this->_trainingAccuracy);
        std::cout << " - val_loss:";
        this->_hiddenLayers[0].getValidationLoss(validationSet, this->_hiddenLayers, this->_validationLoss, this->_validationAccuracy);
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

    saveModel();
    displayGraphs();
}

Model::~Model() {}

std::vector<double> Model::loadModel(std::string modelWeights)
{
    std::ifstream file(modelWeights);
    if (!file.is_open())
        std::cerr << "Error opening file: " << modelWeights << std::endl;

    std::string line, token;
    std::vector<double> biases;

    getline(file, line);
    std::stringstream ss(line);
    while (getline(ss, token, ','))
    {
        biases.push_back(std::stod(token));
    }

    while (getline(file, line))
    {
        std::vector<std::vector<double>> layer;
        std::stringstream ss(line);

        while (getline(ss, token, '['))
        {
            std::vector<double> neuronWeights;
            std::stringstream wss(token);
            while (getline(wss, token, ','))
            {
                if (token.find(']') != std::string::npos)
                {
                    size_t pos = token.find(']');
                    token = token.substr(0, pos);
                }
                if (!token.empty())
                    neuronWeights.push_back(std::stod(token));
            }
            if (!neuronWeights.empty())
                layer.push_back(neuronWeights);
        }
        if (!layer.empty())
            _modelArchitecture.push_back(layer);
    }
    return biases;
}

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

    std::string command = "python3 utils/display_graphs.py \"" + validationLossString + "\" \"" + validationAccuracyString + "\" \"" + trainingLossString + "\" \"" + trainingAccuracyString + "\"";
    system(command.c_str());
}

void Model::saveModel()
{
    std::remove("model.txt");

    std::ofstream file("model.txt");

    // first line are the biases of each layer separated by a comma without []
    for (size_t i = 1; i < this->_hiddenLayers.size(); i++)
    {
        file << this->_hiddenLayers[i].getBiasNeuron();
        if (i != this->_hiddenLayers.size() - 1)
            file << ",";
    }
    file << std::endl;

    for (size_t i = 0; i < this->_modelArchitecture.size(); i++)
    {
        file << "[";
        for (size_t j = 0; j < this->_modelArchitecture[i].size(); j++)
        {
            file << "[";
            for (size_t k = 0; k < this->_modelArchitecture[i][j].size(); k++)
            {
                file << this->_modelArchitecture[i][j][k];
                if (k != this->_modelArchitecture[i][j].size() - 1)
                    file << ",";
            }
            file << "]";
            if (j != this->_modelArchitecture[i].size() - 1)
                file << ",";
        }
        file << "]";
        if (i != this->_modelArchitecture.size() - 1)
            file << ",";
        file << std::endl;
    }
}