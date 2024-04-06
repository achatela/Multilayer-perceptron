#include "includes/Model.hpp"

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
    std::vector<std::vector<double>> datas;
    std::ifstream file = fileChecking(filename);
    std::string line, token;
    std::getline(file, line);
    std::stringstream ss(line);
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
    std::vector<std::vector<double>> datas;
    std::ifstream file = fileChecking(filename);
    std::string line, token;
    std::getline(file, line);
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

int main(int argc, char **argv)
{
    try
    {
        if (argc != 3)
        {
            std::cerr << "Usage: ./predict model.txt dataset.csv" << std::endl;
            return 1;
        }
        std::vector<std::vector<double>> predictionSet;
        try
        {
            predictionSet = loadDataset(argv[2]);
        }
        catch (const std::exception &e)
        {
            if (std::string(e.what()) == "stof")
                predictionSet = loadDatasetEvaluation(argv[2]);
        }

        std::ifstream file(argv[2]);
        if (!file.is_open())
        {
            std::cerr << "Error opening file: " << argv[2] << std::endl;
            return 1;
        }

        size_t reference = predictionSet[0].size();
        for (size_t i = 0; i < predictionSet.size(); i++)
        {
            if (predictionSet[i].size() != reference)
            {
                std::cerr << "Error: input dataset is not consistent" << std::endl;
                exit(1);
            }
        }
        Model model(std::string(argv[1]), predictionSet);
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    return 0;
}