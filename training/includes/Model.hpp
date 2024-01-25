#include <vector>
#include "Layer.hpp"

class Model {

    public:
        Model(
            int layerNumbers,
            int epochs
        );
        ~Model();


    private:
        std::vector<Layer> _layers;

}