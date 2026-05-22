#pragma once

#include <cmath>
#include <fstream>
#include <iomanip>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

class NeuralNetwork {
public:
    // A Vector is just a list of numbers, like inputs or neuron outputs.
    using Vector = std::vector<double>;

    NeuralNetwork() = default;

    explicit NeuralNetwork(const std::vector<size_t>& layer_sizes, unsigned seed = std::random_device{}()) {
        reset(layer_sizes, seed);
    }

    void reset(const std::vector<size_t>& layer_sizes, unsigned seed = std::random_device{}()) {
        if (layer_sizes.size() < 2) {
            throw std::invalid_argument("A neural network needs at least an input layer and an output layer.");
        }

        for (size_t size : layer_sizes) {
            if (size == 0) {
                throw std::invalid_argument("Layer sizes must be greater than zero.");
            }
        }

        layer_sizes_ = layer_sizes;
        weights_.clear();
        biases_.clear();

        // The seed lets us get the same random starting values again.
        std::mt19937 generator(seed);
        std::uniform_real_distribution<double> distribution(-1.0, 1.0);

        // Create weights and biases between each pair of layers.
        for (size_t layer = 1; layer < layer_sizes_.size(); ++layer) {
            const size_t input_count = layer_sizes_[layer - 1];
            const size_t output_count = layer_sizes_[layer];

            std::vector<Vector> layer_weights(output_count, Vector(input_count));
            Vector layer_biases(output_count);

            for (size_t output = 0; output < output_count; ++output) {
                for (size_t input = 0; input < input_count; ++input) {
                    // Each connection starts with a small random weight.
                    layer_weights[output][input] = distribution(generator);
                }
                // Each neuron also gets one bias value.
                layer_biases[output] = distribution(generator);
            }

            weights_.push_back(layer_weights);
            biases_.push_back(layer_biases);
        }
    }

    Vector predict(const Vector& input) const {
        ensureReady();
        if (input.size() != layer_sizes_.front()) {
            throw std::invalid_argument("Input size does not match the network input layer.");
        }

        // Activations are the current values moving through the network.
        Vector activations = input;

        for (size_t layer = 0; layer < weights_.size(); ++layer) {
            Vector next(biases_[layer].size());

            for (size_t output = 0; output < weights_[layer].size(); ++output) {
                // Start with the bias, then add every input * weight.
                double weighted_sum = biases_[layer][output];

                for (size_t input_index = 0; input_index < weights_[layer][output].size(); ++input_index) {
                    weighted_sum += weights_[layer][output][input_index] * activations[input_index];
                }

                // Sigmoid squashes the number into the 0 to 1 range.
                next[output] = sigmoid(weighted_sum);
            }

            activations = next;
        }

        return activations;
    }

    double train(const Vector& input, const Vector& target, double learning_rate) {
        ensureReady();
        if (input.size() != layer_sizes_.front()) {
            throw std::invalid_argument("Input size does not match the network input layer.");
        }
        if (target.size() != layer_sizes_.back()) {
            throw std::invalid_argument("Target size does not match the network output layer.");
        }

        // Forward pass: remember every layer's output so backpropagation can reuse it.
        std::vector<Vector> activations;
        activations.push_back(input);

        for (size_t layer = 0; layer < weights_.size(); ++layer) {
            Vector next(biases_[layer].size());

            for (size_t output = 0; output < weights_[layer].size(); ++output) {
                // This is the same forward-pass math used by predict().
                double weighted_sum = biases_[layer][output];

                for (size_t input_index = 0; input_index < weights_[layer][output].size(); ++input_index) {
                    weighted_sum += weights_[layer][output][input_index] * activations[layer][input_index];
                }

                next[output] = sigmoid(weighted_sum);
            }

            activations.push_back(next);
        }

        double loss = 0.0;
        std::vector<Vector> deltas(weights_.size());
        const Vector& output_activations = activations.back();
        deltas.back() = Vector(output_activations.size());

        // Start backpropagation at the output layer.
        for (size_t output = 0; output < output_activations.size(); ++output) {
            const double error = output_activations[output] - target[output];
            loss += 0.5 * error * error;
            // Delta means "how much this neuron affected the error."
            deltas.back()[output] = error * sigmoidDerivativeFromActivation(output_activations[output]);
        }

        // Move backward through hidden layers and calculate their deltas.
        for (int layer = static_cast<int>(weights_.size()) - 2; layer >= 0; --layer) {
            const Vector& layer_activations = activations[static_cast<size_t>(layer) + 1];
            deltas[static_cast<size_t>(layer)] = Vector(layer_activations.size());

            for (size_t neuron = 0; neuron < layer_activations.size(); ++neuron) {
                double downstream_error = 0.0;

                for (size_t next_neuron = 0; next_neuron < deltas[static_cast<size_t>(layer) + 1].size(); ++next_neuron) {
                    // A hidden neuron is blamed based on the neurons it feeds into.
                    downstream_error += weights_[static_cast<size_t>(layer) + 1][next_neuron][neuron] *
                        deltas[static_cast<size_t>(layer) + 1][next_neuron];
                }

                deltas[static_cast<size_t>(layer)][neuron] =
                    downstream_error * sigmoidDerivativeFromActivation(layer_activations[neuron]);
            }
        }

        // Update every weight and bias to make future predictions less wrong.
        for (size_t layer = 0; layer < weights_.size(); ++layer) {
            for (size_t output = 0; output < weights_[layer].size(); ++output) {
                for (size_t input_index = 0; input_index < weights_[layer][output].size(); ++input_index) {
                    // learning_rate controls how big each training step is.
                    weights_[layer][output][input_index] -=
                        learning_rate * deltas[layer][output] * activations[layer][input_index];
                }

                biases_[layer][output] -= learning_rate * deltas[layer][output];
            }
        }

        return loss;
    }

    void save(const std::string& path) const {
        ensureReady();

        // Save a plain text model file so it is easy to inspect.
        std::ofstream output(path);
        if (!output) {
            throw std::runtime_error("Could not open model file for writing: " + path);
        }

        output << std::setprecision(17);
        output << "NN_FROM_SCRATCH_V1\n";
        output << layer_sizes_.size() << "\n";

        for (size_t layer_size : layer_sizes_) {
            output << layer_size << " ";
        }
        output << "\n";

        for (size_t layer = 0; layer < weights_.size(); ++layer) {
            output << weights_[layer].size() << " " << weights_[layer][0].size() << "\n";

            for (const Vector& row : weights_[layer]) {
                for (double weight : row) {
                    output << weight << " ";
                }
                output << "\n";
            }

            output << biases_[layer].size() << "\n";
            for (double bias : biases_[layer]) {
                output << bias << " ";
            }
            output << "\n";
        }
    }

    static NeuralNetwork load(const std::string& path) {
        // Load the same text format written by save().
        std::ifstream input(path);
        if (!input) {
            throw std::runtime_error("Could not open model file for reading: " + path);
        }

        std::string header;
        input >> header;
        if (header != "NN_FROM_SCRATCH_V1") {
            throw std::runtime_error("Unsupported model file format: " + path);
        }

        size_t layer_count = 0;
        input >> layer_count;

        std::vector<size_t> layer_sizes(layer_count);
        for (size_t layer = 0; layer < layer_count; ++layer) {
            input >> layer_sizes[layer];
        }

        NeuralNetwork network;
        network.layer_sizes_ = layer_sizes;
        network.weights_.resize(layer_count - 1);
        network.biases_.resize(layer_count - 1);

        for (size_t layer = 0; layer + 1 < layer_count; ++layer) {
            size_t rows = 0;
            size_t columns = 0;
            input >> rows >> columns;

            if (rows != layer_sizes[layer + 1] || columns != layer_sizes[layer]) {
                throw std::runtime_error("Model file shape does not match the layer sizes.");
            }

            network.weights_[layer] = std::vector<Vector>(rows, Vector(columns));
            for (size_t row = 0; row < rows; ++row) {
                for (size_t column = 0; column < columns; ++column) {
                    input >> network.weights_[layer][row][column];
                }
            }

            size_t bias_count = 0;
            input >> bias_count;
            if (bias_count != rows) {
                throw std::runtime_error("Model file bias count does not match the layer size.");
            }

            network.biases_[layer] = Vector(bias_count);
            for (size_t bias = 0; bias < bias_count; ++bias) {
                input >> network.biases_[layer][bias];
            }
        }

        return network;
    }

private:
    // Example: {2, 4, 1} means 2 inputs, 4 hidden neurons, 1 output.
    std::vector<size_t> layer_sizes_;
    // weights_[layer][neuron][input] stores one connection strength.
    std::vector<std::vector<Vector>> weights_;
    // biases_[layer][neuron] stores one bias for each neuron.
    std::vector<Vector> biases_;

    static double sigmoid(double value) {
        // Sigmoid turns any number into a value between 0 and 1.
        return 1.0 / (1.0 + std::exp(-value));
    }

    static double sigmoidDerivativeFromActivation(double activation) {
        // The derivative tells training how sensitive sigmoid is here.
        return activation * (1.0 - activation);
    }

    void ensureReady() const {
        if (layer_sizes_.empty() || weights_.empty() || biases_.empty()) {
            throw std::runtime_error("Neural network has not been initialized.");
        }
    }
};
