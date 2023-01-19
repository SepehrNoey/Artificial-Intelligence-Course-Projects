# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"

        for _ in range(self.iterations):
            iter_values = util.Counter()
            for state in self.mdp.getStates():
                possibleActions = self.mdp.getPossibleActions(state)
                q_values = []
                for action in possibleActions:
                    q_val = 0
                    candidates = self.mdp.getTransitionStatesAndProbs(state, action)
                    for cand_state, cand_prob in candidates:
                        q_val += cand_prob * (self.mdp.getReward(state, action, cand_state)\
                            + self.discount * self.getValue(cand_state))
                    q_values.append(q_val)
                iter_values[state] = max(q_values) if len(q_values) > 0 else 0  # is it true here to return 0 ????
            self.values = iter_values


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        
        q_value = 0
        candidates = self.mdp.getTransitionStatesAndProbs(state, action)
        for cand_state, cand_prob in candidates:
            q_value += cand_prob * (self.mdp.getReward(state, action, cand_state)\
                + self.discount * self.getValue(cand_state))

        return q_value

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"

        if self.mdp.isTerminal(state):
            return None

        possibleActions = self.mdp.getPossibleActions(state)
        q_values = util.Counter()
        for action in possibleActions:
            q_values[action] = self.computeQValueFromValues(state, action)
        return q_values.argMax()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

        states = self.mdp.getStates()
        for i in range(self.iterations):
            state =  states[i % len(states)]
            possibleActions = self.mdp.getPossibleActions(state)
            q_values = []
            for action in possibleActions:
                q_val = 0
                candidates = self.mdp.getTransitionStatesAndProbs(state, action)
                for cand_state, cand_prob in candidates:
                    q_val += cand_prob * (self.mdp.getReward(state, action, cand_state)\
                        + self.discount * self.getValue(cand_state))
                q_values.append(q_val)
            self.values[state] = max(q_values) if len(q_values) > 0 else 0


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

        from util import PriorityQueue
        states = self.mdp.getStates()
        stateSuccessors = dict()
        queue = PriorityQueue()

        for state in states:
            if not self.mdp.isTerminal(state):
                for action in self.mdp.getPossibleActions(state):
                    for cand_state, _ in self.mdp.getTransitionStatesAndProbs(state, action):
                        if cand_state in stateSuccessors:
                            stateSuccessors[cand_state].add(state)
                        else:
                            stateSuccessors[cand_state] = set()
                            stateSuccessors[cand_state].add(state)

        for state in states:
            qList = []
            if not self.mdp.isTerminal(state):
                for action in self.mdp.getPossibleActions(state):
                    qList.append(self.getQValue(state, action))
                diff = abs(self.values[state] - max(qList))
                queue.push(state, diff * -1)

        for _ in range(self.iterations):
            if queue.isEmpty():
                break
            state = queue.pop()
            if not self.mdp.isTerminal(state):
                qList = []
                for action in self.mdp.getPossibleActions(state):
                    qList.append(self.getQValue(state, action))
                self.values[state] = max(qList)

            for s in stateSuccessors[state]:
                if not self.mdp.isTerminal(state):
                    qList = []
                    for action in self.mdp.getPossibleActions(s):
                        qList.append(self.getQValue(s, action))
                    diff = abs(self.values[s] - max(qList))

                    if diff > self.theta:
                        queue.update(s, diff * -1)
