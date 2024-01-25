#include <vector>
#include "Neuron.hpp"

class Layer {

    public:
        Layer();
        ~Layer();


    private:
        std::vector<Neuron> _neurons;
}