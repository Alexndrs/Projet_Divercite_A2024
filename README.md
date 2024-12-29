# Polytechnique Montréal INF 8175 : Projet Divercité

### Authors
- Alexandre Dréan
- Julien Segonne

### December 2024

## Introduction
This project is part of the INF1875 course and involves the implementation of an intelligent agent for the Divercité game. The [report](./rapport_final.pdf) covers the algorithm used, analysis of results, and potential improvements. The core of our work is in this [file](./Divercite/my_player.py)

## What We Have Done
We implemented an intelligent agent using the MiniMax algorithm with Alpha-Beta pruning to optimize decision-making in the game. We also incorporated heuristics to evaluate game states effectively and limit the branching factor to improve computational efficiency.

## What We Have Learned
- **Algorithm Selection**: MiniMax is effective for adversarial search problems, but it requires optimization techniques like Alpha-Beta pruning to be computationally feasible.
- **Heuristics**: Developing custom heuristics based on game-specific strategies can significantly enhance the performance of the agent.
- **Depth and branching factor Management**: Adjusting the depth and branching factor of the search tree based on the stage of the game is crucial for balancing performance and decision quality.

## Limits We Faced
- **Branching Factor**: The high branching factor of the game made it challenging to explore all possible states, necessitating the use of pruning and state limitation.
- **Computational Constraints**: Limited computational resources restricted our ability to explore deeper game states extensively.
- **Time Constraints**: We could not implement all the ideas we had, such as transposition tables and symmetry exploitation, due to time limitations.

## Skills Acquired
- **Algorithm Implementation**: Gained experience in implementing and optimizing adversarial search algorithms.
- **Heuristic Development**: Learned how to develop and fine-tune heuristics for evaluating game states.
- **Optimization Techniques**: Applied various optimization techniques like Alpha-Beta pruning and heuristic use to improve algorithm efficiency.
- **Project Management**: Enhanced our ability to manage a project, prioritize tasks, and work within time constraints.

## Conclusion
Our intelligent agent aims to maximize computational efficiency and decision quality by using MiniMax with Alpha-Beta pruning and custom heuristics. While we faced several challenges, we successfully implemented a competitive agent and identified areas for future improvement.
