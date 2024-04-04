#include "includes/Model.hpp"

int main(int argc, char **argv)
{
    std::vector<std::vector<double>> predictionSet;
    std::ifstream file(argv[2]);
    if (!file.is_open())
    {
        std::cerr << "Error opening file: " << argv[2] << std::endl;
        return 1;
    }

    std::string line, token;
    std::getline(file, line);
    std::stringstream ss(line);
    while (std::getline(file, line))
    {
        std::vector<double> row;
        std::stringstream ss(line);
        while (std::getline(ss, token, ','))
            row.push_back(std::stof(token));
        predictionSet.push_back(row);
    }
    Model model(std::string(argv[1]), predictionSet);
    return 0;
}