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

std::vector<std::vector<double>> parseCsv(const std::string &filename)
{
    std::vector<std::vector<double>> data;
    std::ifstream file = fileChecking(filename);
    std::string line;

    while (getline(file, line))
    {
        double value;
        std::vector<double> row;
        std::stringstream ss(line);

        while (ss >> value)
        {
            row.push_back(value);
            if (ss.peek() == ',')
                ss.ignore();
        }

        data.push_back(row);
    }
    file.close();
    return data;
}

int main(int argc, char **argv)
{
    // std::cout.precision(100);
    if (argc != 6)
    {
        std::cerr << "Usage: " << argv[0] << " \"input file\" \"training_dataset\"  \"n of epochs\" \"learning_rate\" \"hidden layers pattern like 16 8\"" << std::endl;
        return 1;
    }

    // std::vector<std::vector<double>> inputs = parseCsv(argv[1]);
    std::vector<std::vector<double>> validationSet = parseCsv(argv[2]);

    std::ifstream file = fileChecking(argv[1]);
    std::vector<std::vector<double>> inputs;
    std::vector<std::string> columnNames;
    std::string line;
    std::getline(file, line);
    std::string token;
    std::stringstream ss(line);
    while (std::getline(ss, token, ',')) // TODO this skips the first line
    {
        columnNames.push_back(token);
    }
    while (std::getline(file, line))
    {
        std::vector<double> row;
        std::stringstream ss(line);
        while (std::getline(ss, token, ','))
            row.push_back(std::stof(token));
        inputs.push_back(row);
    }
    file.close();

    // std::vector<std::vector<double>> inputs;
    // std::ifstream file = fileChecking(argv[1]);
    // std::string line;
    // while (std::getline(file, line))
    // {
    //     std::vector<double> row;
    //     std::stringstream ss(line);
    //     std::string token;
    //     while (std::getline(ss, token, ','))
    //         row.push_back(std::stof(token));
    //     inputs.push_back(row);
    // }

    std::vector<double> hiddenLayersPattern;
    std::string tokenHidden;
    std::stringstream ssHidden(argv[5]);
    while (std::getline(ssHidden, tokenHidden, ' '))
    {
        hiddenLayersPattern.push_back(std::stof(tokenHidden));
    }

    std::cout << "Inputs: " << inputs.size() << " Validation set: " << validationSet.size() << std::endl;

    Model model(inputs, validationSet, atoi(argv[3]), atof(argv[4]), hiddenLayersPattern);
}