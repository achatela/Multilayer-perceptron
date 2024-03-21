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
    if (argc != 5)
    {
        std::cerr << "Usage: " << argv[0] << " <input file> <training_dataset> <n of epochs> <learning_rate>" << std::endl;
        return 1;
    }
    std::ifstream file = fileChecking(argv[1]);
    std::vector<std::vector<double>> inputs;
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
        std::vector<double> row;
        std::stringstream ss(line);
        while (std::getline(ss, token, ','))
        {
            if (token == "B" || token == "M")
            {
                if (token == "M")
                    row.push_back(0);
                else
                    row.push_back(1);
                continue;
            }
            row.push_back(std::stof(token));
        }
        inputs.push_back(row);
    }

    std::vector<std::vector<double>> validationSet;
    if (argc == 5)
    {
        std::ifstream file = fileChecking(argv[2]);
        std::string line;
        while (std::getline(file, line))
        {
            std::vector<double> row;
            std::stringstream ss(line);
            std::string token;
            while (std::getline(ss, token, ','))
            {
                if (token == "B" || token == "M")
                {
                    if (token == "M")
                        row.push_back(0);
                    else
                        row.push_back(1);
                    continue;
                }
                row.push_back(std::stof(token));
            }
            validationSet.push_back(row);
        }
    }
    std::cout << "Validation set size: " << validationSet.size() << std::endl;
    Model model(inputs, columnNames, validationSet, 2, atoi(argv[3]), atof(argv[4]));
}