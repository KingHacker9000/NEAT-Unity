# NEAT-Unity Integration

This project provides an open-source Python implementation of the NEAT (NeuroEvolution of Augmenting Topologies) algorithm, integrated with Unity to allow developers to create and evolve neural networks in a Unity-based environment. NEAT is a popular evolutionary algorithm designed for generating complex artificial neural networks by optimizing both their structure and weights. This repository serves as a tool for developers and researchers interested in creating AI agents with evolving behaviors in Unity, ideal for projects in AI, simulations, and game development.

## NOTE: THIS PROJECT IS STILL IN DEVELOPMENT

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Unity Integration](#unity-integration)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

---

## Features
- **Modular Python NEAT Implementation**: Allows for customization of mutation rates, fitness functions, and network topology.
- **Unity Integration**: Communicate between Python and Unity through a socket interface to enable real-time training and evolution of agents.
- **Real-Time Visualization**: View your AI agents learning and evolving directly in Unity.
- **Customizable Configurations**: Easily adjustable parameters for NEAT evolution and Unity environment setup.
- **Cross-Platform Compatibility**: Compatible with Windows, MacOS, and Linux.

## Installation

### Requirements
- **Python**: >= 3.7
- **Unity**: 2020.3 or newer
- **NEAT-Python**: `pip install neat-python`
- **Socket communication libraries** (Python and Unity compatible, e.g., `socket` in Python)

### Setup

1. **Clone this repository:**
   ```bash
   git clone https://github.com/KingHacker9000/NEAT-Unity.git
   cd NEAT-Unity
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Import the Unity Package**:
   - Open your Unity project.
   - Import the provided `NEAT-Unity.unitypackage` located in the `Unity/` directory.

## Usage

### 1. Python NEAT Training
   To train neural networks in Python, modify the `train.py` file to set your desired NEAT parameters and start training. Adjust the fitness function based on your specific task.

   ```bash
   python train.py
   ```

### 2. Unity Environment Setup
   In Unity, set up the environment where the agents will be trained. This includes:
   - **Agent Prefabs**: Configure agents with sensors and controllers to interact with the environment.
   - **Environment Scripts**: Use the provided scripts in `Unity/Scripts/` to manage communication between Unity and the Python backend.

### 3. Communication
   Start the socket server in Python, which will communicate with Unity in real-time to evaluate agent performance and update the environment based on NEAT results.

   ```bash
   python socket_server.py
   ```

### 4. Running the Simulation
   - Launch the Python server.
   - Start the Unity scene. Unity will connect to the Python NEAT server and begin evolving agents in real-time.

## Configuration
The NEAT algorithm parameters are configured in `config-feedforward.txt`. Key parameters include:

- `pop_size`: The population size of each generation.
- `generations`: Total number of generations to evolve.
- `mutation_rate`: Mutation rate for connections.
- `fitness_threshold`: Threshold for fitness at which evolution stops.

To adjust the Unity simulation settings, edit the environment parameters in Unity's inspector panel for the environment game object.

## Unity Integration
This project uses a socket connection to facilitate real-time communication between Unity and Python. 

1. **Python to Unity**: The Python NEAT algorithm sends neural network data (weights, activation functions) to Unity for evaluation.
2. **Unity to Python**: Unity calculates the agent's fitness score based on its performance in the environment and sends it back to Python.

Ensure that the Unity environment is prepared to handle agent resets and data updates for real-time NEAT evolution.

## Examples
Explore examples in the `examples/` folder to see configurations and demonstrations of how to set up different tasks such as:
- Evolving agents to navigate a maze.
- Training bots to compete in simple obstacle courses.
- Implementing competitive AI using NEAT to adapt in adversarial environments.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your improvements. Whether it's bug fixes, new features, or better documentation, your help is appreciated!

1. Fork this repository.
2. Create a new branch with your feature or bugfix (`git checkout -b feature-name`).
3. Commit your changes (`git commit -am 'Add a new feature'`).
4. Push to the branch (`git push origin feature-name`).
5. Open a Pull Request.

### Project has not been completed, There may be missing files!

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
