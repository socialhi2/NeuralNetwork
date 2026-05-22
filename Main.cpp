#include "NeuralNetwork.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

struct TrainingExample {
    // One training example is a question and the answer we want.
    std::vector<double> input;
    std::vector<double> target;
};

bool fileExists(const std::string& path) {
    // Try opening the file. If it opens, it exists.
    std::ifstream file(path);
    return file.good();
}

int main() {
    try {
        // These values control one training run.
        const std::string model_path = "xor_model.nn";
        const double learning_rate = 0.5;
        const int epochs = 10000;

        NeuralNetwork network;
        if (fileExists(model_path)) {
            // Continue training from the last saved model.
            network = NeuralNetwork::load(model_path);
            std::cout << "Loaded existing model from " << model_path << ".\n";
        } else {
            // Create a new network: 2 inputs, 4 hidden neurons, 1 output.
            network = NeuralNetwork({2, 4, 1}, 42);
            std::cout << "Created a new 2-4-1 neural network.\n";
        }

        // XOR training data. The network should learn these four answers.
        const std::vector<TrainingExample> examples = {
            {{0.0, 0.0}, {0.0}},
            {{0.0, 1.0}, {1.0}},
            {{1.0, 0.0}, {1.0}},
            {{1.0, 1.0}, {0.0}},
        };

        // One epoch means the network sees every training example once.
        for (int epoch = 1; epoch <= epochs; ++epoch) {
            double total_loss = 0.0;

            for (const TrainingExample& example : examples) {
                // Train on one example and add its error to the total.
                total_loss += network.train(example.input, example.target, learning_rate);
            }

            if (epoch == 1 || epoch % 2000 == 0) {
                // Loss is how wrong the network is. Smaller is better.
                std::cout << "Epoch " << std::setw(5) << epoch
                    << " | average loss: " << total_loss / examples.size() << "\n";
            }
        }

        std::cout << "\nPredictions after training:\n";
        std::cout << std::fixed << std::setprecision(4);

        for (const TrainingExample& example : examples) {
            // Predict uses the trained weights without changing them.
            const std::vector<double> prediction = network.predict(example.input);
            std::cout << example.input[0] << " xor " << example.input[1]
                << " = " << prediction[0] << "\n";
        }

        // Save the learned weights and biases for the next run.
        network.save(model_path);
        std::cout << "\nSaved model to " << model_path << ". Run the program again to keep training it.\n";
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << "\n";
        return 1;
    }

    return 0;
}
