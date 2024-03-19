#include "../includes/Model.hpp"

Model::Model(std::vector<std::vector<double>> inputs, std::vector<std::string> columnNames, std::vector<std::vector<double>> validationSet, int hiddenLayersNumber = 2, int epochs = 100, double learningRate = 0.1) : _inputLayer(inputs), _outputLayer(Layer(2, this->_inputLayer.size(), columnNames.size(), true)), _columnNames(columnNames), _epochs(epochs)
{
    int neuronsNumber = 4;
    this->_hiddenLayers.push_back(Layer(this->_inputLayer));
    int weightsNumber = columnNames.size();
    for (int i = 0; i < hiddenLayersNumber; i++)
    {
        this->_hiddenLayers.push_back(Layer(neuronsNumber, neuronsNumber * 2, this->_columnNames.size(), weightsNumber));
        weightsNumber = neuronsNumber;
        neuronsNumber /= 2;
    }
    this->_hiddenLayers.push_back(Layer(2, weightsNumber, this->_columnNames.size(), weightsNumber, true)); // TODO change 2 to be the number of classes detected in the dataset
    // this->_hiddenLayers.push_back(this->_outputLayer);
    setClassesInputs(inputs);
    for (int i = 0; i < epochs; i++)
    {
        for (std::vector<double> input : inputs)
        {
            std::vector<double> networkWeights;
            for (size_t j = 1; j < this->_hiddenLayers.size(); j++)
            {
                if (j == this->_hiddenLayers.size() - 1)
                {
                    this->_hiddenLayers[j].feedForward(this->_hiddenLayers[j - 1], 2, this->_inputLayer, input, networkWeights);
                }
                else if (j == 1)
                    this->_hiddenLayers[j].feedForward(this->_hiddenLayers[j - 1], 0, this->_inputLayer, input);
                else
                    this->_hiddenLayers[j].feedForward(this->_hiddenLayers[j - 1], 1, this->_inputLayer, input);

                // auto weights = this->_hiddenLayers[j].getNeurons();
                // for (auto &neuron : weights)
                // {
                //     for (auto &weight : neuron.getWeights())
                //     {
                //         networkWeights.push_back(weight);
                //     }
                // }
            }
            this->_hiddenLayers.back().backPropagation(this->_hiddenLayers, input, learningRate);
        }
        std::vector<double> probabilities;
        for (auto &output : this->_hiddenLayers.back().getNeurons())
            probabilities.push_back(output.getOutput());
        double validation_loss = this->_hiddenLayers.back().getValidationLoss(validationSet, probabilities);
        double loss = this->_hiddenLayers.back().getValidationLoss(inputs, probabilities);
        // double accuracy = this->getAccuracy(inputs, _hiddenLayers);
        std::cout << "epoch " << i + 1 << "/" << epochs << " - loss: " << loss << " - val_loss: " << validation_loss << std::endl;
    }
    std::vector<std::vector<double>> finalWeights;
    // for (auto &output : this->_hiddenLayers.back().getNeurons())
    // {
    //     finalWeights.push_back(output.getWeights());
    // }
    // setFinalWeights(finalWeights);
}

Model::~Model()
{
}

int Model::predictClass(std::vector<double> input, std::vector<Layer> layers)
{
    for (size_t j = 1; j < layers.size(); j++)
    {
        if (j == layers.size() - 1)
            layers[j].feedForward(layers[j - 1], 2, this->_inputLayer, input);
        else if (j == 1)
            layers[j].feedForward(layers[j - 1], 0, this->_inputLayer, input);
        else
            layers[j].feedForward(layers[j - 1], 1, this->_inputLayer, input);
    }
    std::vector<double> probabilities;
    for (auto &output : layers.back().getNeurons())
        probabilities.push_back(output.getOutput());
    return std::distance(probabilities.begin(), std::max_element(probabilities.begin(), probabilities.end()));
}

void Model::setClassesInputs(std::vector<std::vector<double>> inputs)
{
    for (size_t i = 0; i < inputs.size(); i++)
    {
        std::vector<double> tmp;
        if (inputs[i][0] + 1 > _classesInputs.size())
            while (inputs[i][0] + 1 > _classesInputs.size())
                _classesInputs.push_back(std::vector<std::vector<double>>());
        for (size_t j = 1; j < inputs[i].size(); j++) // stqrts at 1 because the first element is the class
        {
            tmp.push_back(inputs[i][j]);
        }
        _classesInputs[inputs[i][0]].push_back(tmp);
    }
}

// double Model::getAccuracy(std::vector<std::vector<double>> inputs, std::vector<Layer> layers)
// {
//     int count = 0;
//     for (auto &input : inputs)
//     {
//         int predictedClass = predictClass(input, layers);
//         if (predictedClass == input[0])
//             count++;
//     }
//     return (double)count / inputs.size();
// }