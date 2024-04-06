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

std::vector<std::vector<double>> loadDataset(std::string filename)
{
    std::ifstream file = fileChecking(filename);
    std::vector<std::vector<double>> datas;
    std::vector<std::string> columnNames;
    std::string line;
    std::getline(file, line);
    std::string token;
    std::stringstream ss(line);

    while (std::getline(ss, token, ',')) // Skip the first line
    {
        columnNames.push_back(token);
    }

    while (std::getline(file, line))
    {
        std::vector<double> row;
        std::stringstream ss(line);
        while (std::getline(ss, token, ','))
            row.push_back(std::stof(token));
        datas.push_back(row);
    }
    file.close();

    return datas;
}

void normalize(std::vector<std::vector<double>> &datas)
{
    for (size_t i = 0; i < datas[0].size(); i++)
    {
        double max = -1;
        double min = 1000000000;
        for (size_t j = 0; j < datas.size(); j++)
        {
            if (datas[j][i] > max)
                max = datas[j][i];
            if (datas[j][i] < min)
                min = datas[j][i];
        }
        for (size_t j = 0; j < datas.size(); j++)
            datas[j][i] = (datas[j][i] - min) / (max - min);
    }
}

std::vector<std::vector<double>> loadDatasetEvaluation(std::string filename)
{
    std::ifstream file = fileChecking(filename);
    std::vector<std::vector<double>> datas;
    std::vector<std::string> columnNames;
    std::string line;
    std::string token;
    std::stringstream ss(line);

    while (std::getline(file, line))
    {
        std::vector<double> row;
        std::stringstream ss(line);
        std::getline(ss, token, ','); // Skip the first column
        while (std::getline(ss, token, ','))
        {
            if (token == "M")
                row.push_back(0);
            else if (token == "B")
                row.push_back(1);
            else
                row.push_back(std::stof(token));
        }
        datas.push_back(row);
    }
    file.close();

    normalize(datas);

    return datas;
}

void checkCorrectness(std::vector<std::vector<double>> &inputs, std::vector<std::vector<double>> &validationSet)
{
    size_t reference = inputs[0].size();
    for (size_t i = 0; i < inputs.size(); i++)
    {
        if (inputs[i].size() != reference)
        {
            std::cerr << "Error: input dataset is not consistent" << std::endl;
            exit(1);
        }
    }

    for (size_t i = 0; i < validationSet.size(); i++)
    {
        if (validationSet[i].size() != reference)
        {
            std::cerr << "Error: validation dataset is not consistent" << std::endl;
            exit(1);
        }
    }
}

int main(int argc, char **argv)
{
    try
    {
        if (argc != 6 && argc != 3)
        {
            std::cerr << "Usage: " << argv[0] << " \"input file\" \"training_dataset\"  \"n of epochs\" \"learning_rate\" \"hidden layers pattern like 16 8\"" << std::endl;
            return 1;
        }

        std::vector<std::vector<double>> inputs;
        std::vector<std::vector<double>> validationSet;

        try
        {
            inputs = loadDataset(std::string(argv[1]));
            validationSet = loadDataset(std::string(argv[2]));
        }
        catch (const std::exception &e) // To comply with the dataset produced by evaluation.py
        {
            if (std::string(e.what()) == "stof")
            {
                inputs = loadDatasetEvaluation(std::string(argv[1]));
                validationSet = loadDatasetEvaluation(std::string(argv[2]));
            }
            else
            {
                std::cerr << e.what() << std::endl;
                throw;
            }
        }

        checkCorrectness(inputs, validationSet);

        if (argc == 3)
            Model model(inputs, validationSet);
        else if (argc == 6)
        {
            std::vector<double> hiddenLayersPattern;
            std::string tokenHidden;
            std::stringstream ssHidden(argv[5]);
            while (std::getline(ssHidden, tokenHidden, ' '))
            {
                hiddenLayersPattern.push_back(std::stof(tokenHidden));
            }
            Model model(inputs, validationSet, atoi(argv[3]), atof(argv[4]), hiddenLayersPattern);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}