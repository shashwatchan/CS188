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
        self.values = {}
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        k = self.iterations
        prevalues = {}
        mdp = self.mdp
        for i in mdp.getStates():
          prevalues[i] = 0

        for _ in range(k):
          newvalues = {}
          for state in mdp.getStates():
            newvalues[state] = None
            for action in mdp.getPossibleActions(state):
              forthisres = 0
              for nextState,prob in mdp.getTransitionStatesAndProbs(state,action):
                forthisres += prob*(self.discount*prevalues[nextState]+mdp.getReward(state,action,nextState));
              if(newvalues[state] == None or forthisres > newvalues[state]):
                newvalues[state] = forthisres
            if(newvalues[state] == None):
              newvalues[state] = 0
          prevalues = newvalues

        self.values = prevalues


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
        res = 0
        mdp = self.mdp
        for nextState,prob in mdp.getTransitionStatesAndProbs(state,action):
          res += prob*(self.discount*self.values[nextState]+mdp.getReward(state,action,nextState));
        return res
        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        bestAction = None
        bestValue = 0
        mdp = self.mdp
        for action in mdp.getPossibleActions(state):
          forthisres = 0
          for nextState,prob in mdp.getTransitionStatesAndProbs(state,action):
                forthisres += prob*(self.discount*self.values[nextState]+mdp.getReward(state,action,nextState));
          if(bestAction == None or forthisres > bestValue):
            bestAction = action
            bestValue = forthisres
        return bestAction
        util.raiseNotDefined()

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
        k = self.iterations
        mdp = self.mdp
        for i in mdp.getStates():
          self.values[i] = 0

        while k > 0:
          for state in mdp.getStates():
            k -= 1  
            newval = None
            for action in mdp.getPossibleActions(state):
              forthisres = 0
              for nextState,prob in mdp.getTransitionStatesAndProbs(state,action):
                forthisres += prob*(self.discount*self.values[nextState]+mdp.getReward(state,action,nextState));
              
              if(newval == None or forthisres > newval):
                newval = forthisres

            if(newval == None):
              newval = 0
            self.values[state]= newval
            if(k == 0):
              break

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
        self.preds = {}
        for i in mdp.getStates():
          self.preds[i] = set();
        for i in mdp.getStates():
          for action in mdp.getPossibleActions(i):
            for nextState,prob in mdp.getTransitionStatesAndProbs(i,action):
              if(prob == 0):
                continue
              self.preds[nextState].add(i)
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        pq = util.PriorityQueue()
        for i in self.mdp.getStates():
          self.values[i] = 0
        for state in self.mdp.getStates():
          hi = None
          for action in self.mdp.getPossibleActions(state):
            qval = self.getQValue(state,action);
            if hi == None or qval > hi:
              hi = qval
          if hi != None:
            pq.push(state,-abs(qval-self.values[state]))

        for i in range(self.iterations):
          if pq.isEmpty():
            break
          state = pq.pop();

          newval = None
          for action in self.mdp.getPossibleActions(state):
            forthisres = 0
            for nextState,prob in self.mdp.getTransitionStatesAndProbs(state,action):
              forthisres += prob*(self.discount*self.values[nextState]+self.mdp.getReward(state,action,nextState));
              
            if(newval == None or forthisres > newval):
              newval = forthisres

          if(newval == None):
            newval = 0
          self.values[state]= newval

          for cands in self.preds[state]:
            hi = None
            for action in self.mdp.getPossibleActions(cands):
              qval = self.getQValue(cands,action);
              if hi == None or qval > hi:
                hi = qval
            diff = abs(hi-self.values[cands])
            if(diff > self.theta):
              pq.update(cands,-diff)




        

