#include <vector>
#include <algorithm>
#include "Neuron.hpp"

class Layer
{

public:
    // for the input layer
    Layer(std::vector<std::vector<double>> inputs);
    // for the hidden layers
    Layer(int neuronsNumber, int sizePreviousLayer, int featureNumber, int weightsNumber);
    // for the output layer
    Layer(int neuronsNumber, int sizePreviousLayer, int featureNumber, int weightsNumber, bool isOutputLayer);
    ~Layer();

    double feedForward(Layer &previousLayer, int mode, std::vector<std::vector<double>> inputs = {}, std::vector<double> input = {}, std::vector<double> networkWeights = {});
    void backPropagation(std::vector<Layer> &layers, std::vector<double> input, double learningRate);

    std::vector<Neuron> &getNeurons() { return this->_neurons; };
    std::vector<double> softmaxFunction(std::vector<double> inputs);

    double getValidationLoss(std::vector<std::vector<double>> validationSet, std::vector<Layer> layers);

    void setLoss(double loss) { this->loss = loss; };
    double getLoss() { return this->loss; };

    void firstHiddenLayerFeed(Layer &previousLayer, std::vector<double> input);
    void hiddenLayerFeed(Layer &previousLayer);
    std::vector<double> outputLayerFeed(Layer &previousLayer);

    void sigmoid(double sum, int i);

    double crossEntropyLoss(std::vector<double> probabilities, int result, std::vector<double> networkWeights);

private:
    double loss;
    std::vector<Neuron> _neurons;
};