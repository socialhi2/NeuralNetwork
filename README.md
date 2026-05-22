# NeuralNetwork

This is a from-scratch C++ neural network learning project.

The current program trains a tiny neural network to solve XOR:

```text
0 xor 0 = 0
0 xor 1 = 1
1 xor 0 = 1
1 xor 1 = 0
```

XOR is a good first target because a straight line cannot solve it. That means the network must use a hidden layer and learn a non-linear pattern.

## What exists now

- `include/NeuralNetwork.hpp` contains a small feed-forward neural network.
- `Main.cpp` trains the network on XOR, prints predictions, and saves the trained model.
- `xor_model.nn` is created when you run the program. If it already exists, the program loads it and keeps training from the saved weights.

## Build and run

From the project folder:

```powershell
cmake -S . -B build
cmake --build build
```

Then run the executable CMake creates. With Visual Studio generators this is commonly:

```powershell
.\build\Debug\NeuralNetwork.exe
```

With single-config generators it may be:

```powershell
.\build\NeuralNetwork.exe
```

If CMake cannot find a C++ compiler, install one of these:

- Visual Studio 2022 Build Tools with the C++ workload
- MinGW-w64
- LLVM/Clang for Windows

## The learning path

1. Understand the data

   Inputs are numbers. Targets are the answers you want the network to learn. For XOR, each input has two numbers and the target has one number.

2. Understand the network shape

   The demo uses `2-4-1`:

   - 2 input neurons
   - 4 hidden neurons
   - 1 output neuron

3. Forward pass

   The network multiplies inputs by weights, adds biases, and passes the result through `sigmoid`.

4. Loss

   Loss measures how wrong the prediction was. Lower loss means the network is improving.

5. Backpropagation

   Backpropagation calculates how each weight and bias contributed to the error.

6. Gradient descent

   The network nudges each weight and bias in the direction that should reduce future error.

7. Save and continue

   `network.save("xor_model.nn")` stores the learned weights and biases. `NeuralNetwork::load(...)` brings them back later so training can continue.

## Important note about continuous learning

This project already supports continued training: load a saved model, train it on data, save it again.

True continuous learning is more than saving and reloading. A model can forget older patterns when trained only on new data. Later milestones should add:

- a replay buffer of older examples
- validation tests to catch forgetting
- learning-rate schedules
- versioned model files
- backups before training on new data

## Exercises

- Change `{2, 4, 1}` in `Main.cpp` to `{2, 8, 1}` and compare the loss.
- Change `learning_rate` from `0.5` to `0.1` and watch training slow down.
- Delete `xor_model.nn` and rerun to start fresh.
- Add a function that trains for one epoch, then call it from `main`.
- Add a CSV loader so the network can learn from a file instead of hard-coded examples.

## Next milestones

- Add command-line options for model path, epochs, and learning rate.
- Add CSV training data support.
- Add mini-batch training.
- Add ReLU activation for hidden layers.
- Add softmax and cross-entropy for multi-class classification.
- Add tests for saving, loading, prediction shape, and XOR learning.
- Use a math library such as Eigen once the from-scratch version is understood.
