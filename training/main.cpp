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

    Model model(inputs, columnNames, 2, 70);
    std::vector<std::vector<std::vector<float>>> classesInputs = model.getClassesInputs();

    int count = 0;
    int zero = 0;
    int one = 0;
    for (int i = 0; i < classesInputs.size(); i++)
    {
        for (int j = 0; j < classesInputs[i].size(); j++)
        {
            int answer = model.predictClass(classesInputs[i][j]);
            if (answer == 0)
                zero++;
            else
                one++;
            if (answer == i)
                count++;
        }
    }
    std::cout << "Zero: " << zero << std::endl;
    std::cout << "One: " << one << std::endl;
    std::cout << "Accuracy: " << (float)count / inputs.size() << std::endl;
}