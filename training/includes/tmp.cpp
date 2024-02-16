#include <iostream>
#include <vector>
#include <cmath>

std::vector<float> softmaxFunction(const std::vector<std::vector<float>> &inputs)
{
    std::vector<float> outputs;

    for (const auto &input : inputs)
    {
        float max_input = *std::max_element(input.begin(), input.end());
        float sum = 0.0;
        std::vector<float> exponentials;

        // Calculate the sum of exponentials
        for (float val : input)
        {
            float exponential = exp(val - max_input);
            exponentials.push_back(exponential);
            sum += exponential;
        }

        // Calculate softmax output
        std::vector<float> softmax_output;
        for (float exp_val : exponentials)
        {
            softmax_output.push_back(exp_val / sum);
        }

        // Append to outputs
        outputs.insert(outputs.end(), softmax_output.begin(), softmax_output.end());
    }

    return outputs;
}

int main()
{
    std::vector<std::vector<float>> inputs = {{1.0, 2.0, 3.0}, {2.0, 5.0, 1.0}};
    std::vector<float> result = softmaxFunction(inputs);

    for (float val : result)
    {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    return 0;
}
