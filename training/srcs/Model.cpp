#include "../includes/Model.hpp"

Model::Model(std::vector<std::vector<double>> inputs, std::vector<std::string> columnNames, std::vector<std::vector<double>> validationSet, int hiddenLayersNumber = 2, int epochs = 100, double learningRate = 0.1) : _inputLayer(inputs), _columnNames(columnNames), _epochs(epochs), _outputLayer(Layer(2, this->_inputLayer.size(), columnNames.size(), true))
{
    int neuronsNumber = 4;
    this->_hiddenLayers.push_back(Layer(this->_inputLayer));
    int weightsNumber = columnNames.size();
    for (int i = 0; i < hiddenLayersNumber; i++)
    {
        this->_hiddenLayers.push_back(Layer(neuronsNumber, weightsNumber, this->_columnNames.size(), weightsNumber));
        weightsNumber = neuronsNumber;
        neuronsNumber /= 2;
    }
    this->_hiddenLayers.push_back(Layer(2, weightsNumber, this->_columnNames.size(), weightsNumber, true)); // TODO change 2 to be the number of classes detected in the dataset
    // this->_hiddenLayers.push_back(this->_outputLayer);
    setClassesInputs(inputs);
    for (int i = 0; i < epochs; i++)
    {
        for (int j = 1; j < this->_hiddenLayers.size(); j++)
        {
            if (j == this->_hiddenLayers.size() - 1)
                this->_hiddenLayers[j].feedForward(this->_hiddenLayers[j - 1], 2, inputs);
            else if (j == 1)
                this->_hiddenLayers[j].feedForward(this->_hiddenLayers[j - 1], 0);
            else
                this->_hiddenLayers[j].feedForward(this->_hiddenLayers[j - 1], 1);
        }
        this->_hiddenLayers.back().backPropagation(this->_hiddenLayers, inputs, learningRate);
        std::vector<std::vector<double>> finalWeights;
        // for (auto &output : this->_hiddenLayers.back().getNeurons())
        // {
        //     finalWeights.push_back(output.getWeights());
        // }
        // double validation_loss = this->_hiddenLayers.back().getValidationLoss(validationSet, finalWeights);
        // std::cout << "epoch " << i + 1 << "/" << epochs << " - loss: " << this->_hiddenLayers.back().getLoss() << " - val_loss:" << validation_loss << std::endl;
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

void Model::setClassesInputs(std::vector<std::vector<double>> inputs)
{
    for (int i = 0; i < inputs.size(); i++)
    {
        std::vector<double> tmp;
        if (inputs[i][0] + 1 > _classesInputs.size())
            while (inputs[i][0] + 1 > _classesInputs.size())
                _classesInputs.push_back(std::vector<std::vector<double>>());
        for (int j = 1; j < inputs[i].size(); j++) // stqrts at 1 because the first element is the class
        {
            tmp.push_back(inputs[i][j]);
        }
        _classesInputs[inputs[i][0]].push_back(tmp);
    }
}

int Model::predictClass(std::vector<double> inputs)
{
    std::vector<std::vector<double>> weights = getFinalWeights();
    std::vector<double> outputs;
    std::vector<double> exponentials;
    double sum = 0.0;

    for (auto &weight : weights)
    {
        double exponential = 0;
        double max_input = 0;
        for (double val : inputs)
        {
            if (val > max_input)
                max_input = val;
        }

        for (int i = 0; i < inputs.size(); i++)
        {
            exponential += exp(inputs[i] * weight[i] - max_input);
        }
        exponentials.push_back(exponential);
        sum += exponential;
    }
    for (int i = 0; i < exponentials.size(); i++)
    {
        outputs.push_back(exponentials[i] / sum);
    }

    int index = 0;
    double max = 0;
    for (int i = 0; i < outputs.size(); i++)
    {
        if (outputs[i] > max)
        {
            max = outputs[i];
            index = i;
        }
    }
    // for (auto &proba : outputs)
    // {
    //     std::cout << proba << " ";
    // }
    // std::cout << std::endl;

    return index;
}