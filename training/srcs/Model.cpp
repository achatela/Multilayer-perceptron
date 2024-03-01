#include "../includes/Model.hpp"

Model::Model(std::vector<std::vector<float>> inputs, std::vector<std::string> columnNames, std::vector<std::vector<float>> validationSet, int hiddenLayersNumber = 2, int epochs = 100, float learningRate = 0.1) : _inputLayer(inputs), _columnNames(columnNames), _epochs(epochs), _outputLayer(Layer(2, this->_inputLayer.size(), columnNames.size(), true))
{
    int neuronsNumber = 4;
    this->_hiddenLayers.push_back(Layer(this->_inputLayer, neuronsNumber));
    for (int i = 0; i < hiddenLayersNumber; i++)
    {
        this->_hiddenLayers.push_back(Layer(neuronsNumber, this->_inputLayer.size(), this->_columnNames.size()));
        neuronsNumber /= 2;
    }
    this->_hiddenLayers.push_back(this->_outputLayer);
    setClassesInputs(inputs);
    for (int i = 0; i < epochs; i++)
    {
        for (int j = 1; j < this->_hiddenLayers.size(); j++)
        {
            if (j == this->_hiddenLayers.size() - 1)
                this->_hiddenLayers[j].feedForward(this->_hiddenLayers[j - 1], 2);
            else if (j == 1)
                this->_hiddenLayers[j].feedForward(this->_hiddenLayers[j - 1], 0);
            else
                this->_hiddenLayers[j].feedForward(this->_hiddenLayers[j - 1], 1);
        }
        this->_hiddenLayers.back().backPropagation(this->_hiddenLayers, inputs, learningRate);
        std::vector<std::vector<float>> finalWeights;
        // for (auto &output : this->_hiddenLayers.back().getNeurons())
        // {
        //     finalWeights.push_back(output.getWeights());
        // }
        float validation_loss = this->_hiddenLayers.back().getValidationLoss(validationSet, finalWeights);
        std::cout << "epoch " << i + 1 << "/" << epochs << " - loss: " << this->_hiddenLayers.back().getLoss() << " - val_loss:" << validation_loss << std::endl;
    }
    std::vector<std::vector<float>> finalWeights;
    // for (auto &output : this->_hiddenLayers.back().getNeurons())
    // {
    //     finalWeights.push_back(output.getWeights());
    // }
    // setFinalWeights(finalWeights);
}

Model::~Model()
{
}

void Model::setClassesInputs(std::vector<std::vector<float>> inputs)
{
    for (int i = 0; i < inputs.size(); i++)
    {
        std::vector<float> tmp;
        if (inputs[i][0] + 1 > _classesInputs.size())
            while (inputs[i][0] + 1 > _classesInputs.size())
                _classesInputs.push_back(std::vector<std::vector<float>>());
        for (int j = 1; j < inputs[i].size(); j++) // stqrts at 1 because the first element is the class
        {
            tmp.push_back(inputs[i][j]);
        }
        _classesInputs[inputs[i][0]].push_back(tmp);
    }
}

int Model::predictClass(std::vector<float> inputs)
{
    std::vector<std::vector<float>> weights = getFinalWeights();
    std::vector<float> outputs;
    std::vector<float> exponentials;
    float sum = 0.0;

    for (auto &weight : weights)
    {
        float exponential = 0;
        float max_input = 0;
        for (float val : inputs)
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
    float max = 0;
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