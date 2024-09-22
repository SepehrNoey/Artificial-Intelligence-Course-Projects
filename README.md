# Principles and Applications of Artificial Intelligence

This repository contains the course projects for the "Principles and Applications of Artificial Intelligence" course. Many of these projects are derived from the [Pacman Projects of the Berkeley AI course](http://ai.berkeley.edu/project_overview.html). These projects cover core AI concepts such as search algorithms, multi-agent systems, reinforcement learning, and probabilistic reasoning.

## Project Demos

### 1. A* Search in Pacman to Find Food
In this project, Pacman uses the A* search algorithm to find the shortest path to collect all food in a maze. The heuristic function helps guide Pacman by estimating the distance to the closest piece of food.

**Demo**:

<video src="https://github.com/user-attachments/assets/a054ed87-c39f-401c-b5bb-4fa5ee167e66">|

### 2. Multi-Agent System with MiniMax and Alpha-Beta Pruning
This project demonstrates a multi-agent solution where Pacman competes against ghosts. The game is solved using the MiniMax algorithm with Alpha-Beta Pruning, allowing the AI to think ahead and make optimal moves up to a depth of 5.

**Demo**:  

<video src="https://github.com/user-attachments/assets/870971dd-4213-4a2c-bb0f-477a06da200c">|

### 3. Q-Learning Demo
This project showcases reinforcement learning with Q-learning. Over 100 iterations, Pacman learns an optimal policy through interaction with the environment and progressively improves its decision-making.

**Demo**:
 
<video src="https://github.com/user-attachments/assets/7a1715bd-a3f0-4f05-85b0-131b4e47ec2c">|

### 4. Crawler Learning to Crawl (Reinforcement Learning)
In this simple game, a virtual crawler learns to move using reinforcement learning techniques. The system rewards the crawler based on its ability to successfully crawl.

**Demo**:  

<video src="https://github.com/user-attachments/assets/968cfe0e-2aed-474f-8787-aa0f1f906eec">|

---

## Concepts and Techniques

The following AI concepts were implemented in these projects:

1. **Search Algorithms**:
   - **Breadth-First Search (BFS)**
   - **Depth-First Search (DFS)**
   - **Uniform Cost Search (UCS)**
   - **A* (A Star) Search**
   - **Greedy Search**

2. **Markov Decision Processes (MDPs)** and **Constraint Satisfaction Problems (CSPs)**:
   - Solving decision-making problems where outcomes are partly random and partly under the control of an agent.
   - Implementing constraints and solving them efficiently using techniques like backtracking and constraint propagation.

3. **Multi-Agent Systems**:
   - **MiniMax Algorithm**: Finding the optimal strategy for agents by minimizing the possible loss.
   - **Alpha-Beta Pruning**: Optimizing MiniMax by pruning branches that don't need to be explored (used with a depth of 5 in the demo).

4. **Reinforcement Learning**:
   - **Value Iteration**: Dynamic programming for computing optimal policies in MDPs.
   - **Q-Learning**: Model-free learning where agents learn to act optimally based on rewards.
   - **Approximate Q-Learning**: Extending Q-learning with function approximation for larger state spaces.

5. **Bayesian Networks**:
   - Probabilistic graphical models representing variables and their conditional dependencies through directed acyclic graphs.