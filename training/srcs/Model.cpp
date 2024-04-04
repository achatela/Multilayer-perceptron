#include "../includes/Model.hpp"

Model::Model(std::string modelWeights)
{
    std::ifstream file(modelWeights);
    if (!file.is_open())
    {
        std::cerr << "Error opening file: " << modelWeights << std::endl;
        return;
    }

    std::string line, token;
    while (getline(file, line))
    {
        std::vector<std::vector<double>> layer;
        std::stringstream ss(line);

        while (getline(ss, token, '['))
        { // Split line into tokens separated by '['
            std::vector<double> neuronWeights;
            std::stringstream wss(token);
            while (getline(wss, token, ','))
            { // Split token into weights separated by ','
                if (token.find(']') != std::string::npos)
                {
                    size_t pos = token.find(']');
                    token = token.substr(0, pos); // Remove ']'
                }
                if (!token.empty())
                {
                    neuronWeights.push_back(std::stod(token)); // Convert string to double and add to vector
                }
            }
            if (!neuronWeights.empty())
            {
                layer.push_back(neuronWeights); // Add neuron weights to layer
            }
        }
        if (!layer.empty())
        {
            _modelArchitecture.push_back(layer); // Add layer to model architecture
        }
    }
}

Model::Model(std::vector<std::vector<double>> &inputs, std::vector<std::string> &columnNames, std::vector<std::vector<double>> &validationSet, int epochs, double learningRate, std::vector<double> &hiddenLayersPattern) : _inputLayer(inputs), _columnNames(columnNames)
{
    this->_hiddenLayers.push_back(Layer(this->_inputLayer));
    double numberWeights = columnNames.size();
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

    // debug print model architecture
    // for (size_t i = 0; i < this->_modelArchitecture.size(); i++)
    // {
    //     std::cout << "Layer " << i << std::endl;
    //     for (size_t j = 0; j < this->_modelArchitecture[i].size(); j++)
    //     {
    //         std::cout << "Neuron " << j << std::endl;
    //         for (size_t k = 0; k < this->_modelArchitecture[i][j].size(); k++)
    //             std::cout << this->_modelArchitecture[i][j][k] << " ";
    //         std::cout << std::endl;
    //     }
    // }

    saveModel();
    displayGraphs(); // block the program until the user closes the graphs for the moment
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

void Model::saveModel()
{
    std::remove("model.txt");

    std::ofstream file("model.txt");

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