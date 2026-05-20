# Overview on Reinforcement Learning With Reward Machines in Stochastic Games

</br>

<table>
  <tr>
  <td bgcolor="#f6f8fa">
    <h3>Project Abstract</h3>
Following the Reinforcement Learning With Reward Machines in Stochastic Games paper <a href="https://arxiv.org/pdf/2305.17372">[1]</a>, we propose an overview of the <i>Q-learning with reward machines for stochastic games</i> (QSG-RM), on the 3 project task. The code, to us originally unavailable, has been reconstructed following the proposed algorithm and full parametrization.
</br></br>
Each task follows the 2-agents Pac-Man game problem structure, where reward functions are assumed to be <b>non-markovian</b> and additional constraints are fixed: fixed cordinates power bases, gridword actions and collisions. The goal of each agent is approximated to the reach its own power base and the consequently capture the other agent. By producing the code structure, we enabled for the generalization to 2-agent  reward-machine dependent problem and game.
</br></br>
Having provided the full reconstruction of task I, II, we try to propose a new complex 2-agent game, defining its relative environment and reward machine, performing ad-hoc reward tuning, showing the behavior. The proposed maze game maintains task complexity and non markovianity, requiring the both agent avoidance, key collection and consequent escape 
</br></br>
  </td>
  </tr>
</table>

</br>

## Code Structure and original problem Formulation

The repository relative code is entirely contained into the `CODE` folder, which is structured by including:

* `simulation_code.py`: _the main file, where the task is executed_
And three files which are (explicilty or implicitly) task specific:
* `environment.py`: _the environment class of the problem (PAC-gridworld), which includes the $L$ function for labelling._
* `reward_machine_task_*.py`: _the RM class, so to get the RMs instances in the simulation file. Here the automata are translated into Python code.
* `game_parameters`: _this file includes global parameters definition for both **problem agnostic** (for the solver) and **problem specific** (agent coordinates etc.) cases._

### Getting ready

The repository code execution relies on the python environment available in `setup.sh` file. Please execute:

```bash
$: ./setup.sh
```

To get the `requirements.txt` satisfied, then activate the environment and use it to the code execution as shown below.

```
$: source QSGRM_env/bin/activate
$: python CODE/simulation_code.py
```

### Generalize to task and user defined problems

The environment and RM python files are task/problem specific. If you need the simulation to operate on a different task, please change the marked line in the `simulation_code.py` file.

```
from reward_machine_task_I import *
```

> Please, be also sure that the functional definition and required functions are present in your modified version.

<br/>

### Observed results and Conclusions



#### Task I

As stated in the original paper in Case Study I, the ego agent is required to first reach its own power base, then destroy/reach the adversarial agent’s power base to be the more powerful, and capture
the adversarial agent afterward. [1]

<p align="center">
  <img src="https://github.com/Matthew6363/Reinforcement-Learning-exam/blob/main/GAME_VISUALIZATION/Exported_gifs/q_models_task_I_ep16000_simulation.gif" width="300" alt="Descrizione GIF">
</p>

> **Task I.** Execution on $6x6$ grid, ['up', 'down', 'left', 'right'] actions, uniform(low=0.0001, high=0.001) Q initialization, game.support_enumeration() nash solver and epsilon decay from 0.3 to 0.05. After 16000 episodes learning visualization.

#### Task II

In Case Study II, the required sequential events for the ego agent to be powerful are: reaching its power base, reaching the adversarial agent’s power base, reaching its power base. These events demonstrate the scenario in that the ego agent first gets energy at its power base, destroys the adversarial agent’s power base using most of its energy, and then gets recharged to capture the adversarial agent.

<p align="center">
  <img src="https://github.com/Matthew6363/Reinforcement-Learning-exam/blob/main/GAME_VISUALIZATION/Exported_gifs/q_models_task_II_ep6500_simulation.gif" width="300" alt="Descrizione GIF">
</p>

> **Task II.** Execution on $6x6$ grid, ['up', 'down', 'left', 'right'] actions, uniform(low=0.0001, high=0.001) Q initialization, game.support_enumeration() nash solver and epsilon decay from 0.3 to 0.05. After 6500 episodes learning visualization.

#### Task III
Case Study III is different from Case Study II in that the adversarial agent randomly samples the starting location from 2 possible locations. [1]

> **Task III.** Execution on $6x6$ grid, ['up', 'down', 'left', 'right'] actions, uniform(low=0.0001, high=0.001) Q initialization, game.support_enumeration() nash solver and epsilon decay from 0.3 to 0.05. After [?] episodes learning visualization.

<br/>

## Proposed maze-based game

The underliying idea of this section was about testing the method capabilities if the game complexity is extended on environmental side. In fact, we do propose the same collision based task, but only for one of the agents: the ego agent is now immersed into a maze $8\times 10$ (instead of $6\times 6$, having same actions available. We can summarize the proposed game rule:

* The ego agent has to **escape the maze** by getting to the only exit;
* The doors do "unlock" only if a key was calleted by the agent on a fixed cell of the gridworld;
* The adv agent behaves as a seeker, which has to *collide* with ego agent to win.

The reward machine for this case can be visualized here:


### Considerations and observed results



### Gameplay visualization

<p align="center">
  <img src="https://github.com/Matthew6363/Reinforcement-Learning-exam/blob/main/GAME_VISUALIZATION_maze/Exported_gifs/q_models_maze_ep4000_simulation.gif" width="300" alt="Descrizione GIF">
</p>

> **Maze game.** Execution on $8x10$ grid, ['up', 'down', 'left', 'right'] actions, uniform(low=0.001, high=0.01) Q initialization, game.support_enumeration() nash solver and epsilon decay from 0.9 to 0.05. After 4000 episodes learning visualization.

<br/>

<p align="center">
  <img src="https://github.com/Matthew6363/Reinforcement-Learning-exam/blob/main/GAME_VISUALIZATION_maze/Exported_gifs/q_models_maze_ep7500_simulation.gif" width="300" alt="Descrizione GIF">
</p>

> **Maze game.** Execution on $8x10$ grid, ['up', 'down', 'left', 'right'] actions, uniform(low=0.001, high=0.01) Q initialization, game.support_enumeration() nash solver and epsilon decay from 0.9 to 0.05. After 6500 episodes learning visualization.

<br/>

<p align="center">
  <img src="https://github.com/Matthew6363/Reinforcement-Learning-exam/blob/main/GAME_VISUALIZATION_maze/Exported_gifs/q_models_maze_ep10000_simulation.gif" width="300" alt="Descrizione GIF">
</p>

> **Maze game.** Execution on $8x10$ grid, ['up', 'down', 'left', 'right'] actions, uniform(low=0.001, high=0.01) Q initialization, game.support_enumeration() nash solver and epsilon decay from 0.9 to 0.05. After 10000 episodes learning visualization.

<br/>

## References

* [1] [Reinforcement Learning With Reward Machines in Stochastic Games](https://arxiv.org/pdf/2305.17372)
