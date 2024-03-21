#include "../includes/Model.hpp"

void debugModelInfos(std::vector<Layer> layers, std::vector<double> input)
{
    std::cout << "Model infos:" << std::endl;
    std::cout << "Input size: " << input.size() << std::endl;
    std::cout << "First hidden layers neurons and weights number: " << layers[1].getNeurons().size() << " " << layers[1].getNeurons()[0].getWeights().size() << std::endl;
    std::cout << "Second hidden layers neurons and weights number: " << layers[2].getNeurons().size() << " " << layers[2].getNeurons()[0].getWeights().size() << std::endl;
    std::cout << "Output layers neurons and weights number: " << layers[3].getNeurons().size() << " " << layers[3].getNeurons()[0].getWeights().size() << std::endl;
}

Model::Model(std::vector<std::vector<double>> inputs, std::vector<std::string> columnNames, std::vector<std::vector<double>> validationSet, int hiddenLayersNumber = 2, int epochs = 100, double learningRate = 0.1) : _inputLayer(inputs), _outputLayer(Layer(2, this->_inputLayer.size(), columnNames.size(), true)), _columnNames(columnNames), _epochs(epochs)
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
    debugModelInfos(this->_hiddenLayers, inputs[0]);
    // this->_hiddenLayers.push_back(this->_outputLayer);
    // setClassesInputs(inputs);
    for (int i = 0; i < epochs; i++)
    {
        double lossSum = 0;
        for (std::vector<double> input : inputs)
        {
            std::vector<double> networkWeights;
            for (size_t j = 1; j < this->_hiddenLayers.size(); j++)
            {
                if (j == this->_hiddenLayers.size() - 1)
                    lossSum += this->_hiddenLayers[j].feedForward(this->_hiddenLayers[j - 1], 2, this->_inputLayer, input, networkWeights);
                else if (j == 1)
                    this->_hiddenLayers[j].feedForward(this->_hiddenLayers[j - 1], 0, this->_inputLayer, input);
                else
                    this->_hiddenLayers[j].feedForward(this->_hiddenLayers[j - 1], 1, this->_inputLayer, input);
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