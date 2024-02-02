#include "includes/Model.hpp"
#include <iostream>
#include <fstream>
#include <sstream>

std::ifstream fileChecking(const std::string filename)
{
    if (filename.find(".csv") == std::string::npos)
    {
        std::cerr << "Error: file is not a .csv file" << std::endl;
        exit(1);
    }
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error: file not found" << std::endl;
        exit(1);
    }
    return file;
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <input file>" << std::endl;
        return 1;
    }
    std::ifstream file = fileChecking(argv[1]);
    std::vector<std::vector<float>> inputs;
    std::vector<std::string> columnNames;
    std::string line;
    std::getline(file, line);
    std::string token;
    std::stringstream ss(line);
    while (std::getline(ss, token, ','))
    {
        columnNames.push_back(token);
    }
    while (std::getline(file, line))
    {
        std::vector<float> row;
        std::stringstream ss(line);
        while (std::getline(ss, token, ','))
        {
            if (token == "B" || token == "M")
            {
                if (token == "B")
                    row.push_back(-1);
                else
                    row.push_back(-2);
                continue;
            }
            row.push_back(std::stof(token));
        }
        inputs.push_back(row);
    }

    Model model(inputs, columnNames, 2, 100);
    auto inputLayer = model.getInputLayer();
    for (auto row : inputLayer)
    {
        for (auto value : row)
        {
            std::cout << value << " ";
        }
        std::cout << std::endl;
        break;
    }
}