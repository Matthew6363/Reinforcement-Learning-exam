# Overview on Reinforcement Learning With Reward Machines in Stochastic Games

</br>

<table>
  <tr>
  <td bgcolor="#f6f8fa">
    <h3>Project Abstract</h3>
Following the Reinforcement Learning With Reward Machines in Stochastic Games paper <a href="https://arxiv.org/pdf/2305.17372">[1]</a>, we propose an overview of the <i>Q-learning with reward machines for stochastic games</i> (QSG-RM), on the 3 project task. The code, to us originally unavailable, has been reconstructed following the proposed algorithm and full parametrization.
</br></br>
Each task follows the 2-agents Pac-Man game problem structure, where reward functions are assumed to be <b>non-markovian</b> and additional constraints are fixed: fixed cordinates power bases, gridword actions and collisions. The goal of each agent is approximated to the reach its own power base and the consequently capture the other agent. By producing the code structure, we enabled for the generalization to 2-agent  reward-machine dependent problem and game.

Having provided the full reconstruction of task I, II, we try to propose a new trial game with relative environment, game parameters and reward machines automata.
</br></br>
  </td>
  </tr>
</table>

</br>

## Code Structure and Problem Formulation

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

## Observed results and Conclusions

### Task I

<img src="https://github.com/Matthew6363/Reinforcement-Learning-exam/blob/main/EXPORT/task_I_windowed%202%20(random%20init).png" width="45%" />

> **Task I.** Execution on $6x6$ grid, ['up', 'down', 'left', 'right'] actions, uniform(low=0.0001, high=0.001) Q initialization, game.support_enumeration() nash solver and epsilon decay from 0.3 to 0.05.

<br/>

## References

* [1] [Reinforcement Learning With Reward Machines in Stochastic Games](https://arxiv.org/pdf/2305.17372)
