# multiAgents.py
# --------------
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


import datetime
from util import manhattanDistance
from game import Directions
import random, util
from game import Agent
from enum import Enum

path = []

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"
        from multiAgents import path
        path.append(gameState.generatePacmanSuccessor(legalMoves[chosenIndex]).getPacmanPosition())

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        from sys import maxsize
        from multiAgents import path
        from statistics import pvariance, mean
        from random import uniform
        from math import sqrt
        currFoodList = currentGameState.getFood().asList()
        newFoodList = newFood.asList()
        currPos = currentGameState.getPacmanPosition()

        # order of badness: dieing > going to visited places > being close to ghosts
        # order of goodness: eating dots > eating scared ghosts
        # now , we will find the closest not eaten dot
        currManList = getManhattanList(currPos, currFoodList)
        minDist = min(currManList)
        minIdx = currManList.index(minDist)
        newManList = getManhattanList(newPos, newFoodList)
        currGhostStates = currentGameState.getGhostStates()
        ghostManList = getManhattanList(newPos, [ghost.getPosition() for ghost in newGhostStates])
        conflicts = False
        for (currGhost, newGhost) in zip(currGhostStates, newGhostStates):
            if currGhost.getPosition() == newPos or newGhost.getPosition() == newPos:
                conflicts = True
                break

        if not conflicts: 
            if len(path) > 0 and newPos == path[-1]:
                return -maxsize # we shouldn't stop, but its badness is less than dieing
            
            # we are getting closer to dot -> so it's good! -> maximum score
            goodWay = 1 if minIdx >= len(newManList) or newManList[minIdx] < currManList[minIdx] else 0
            indexPlusOne = 0
            score = 0
            if newPos in path:
                indexPlusOne = max(index for index, item in enumerate(path) if item == newPos) + 1
            
            condition = len(ghostManList) > 0
            ghostMeanDist = mean(ghostManList) if condition else 0
            ghostPvar = pvariance(ghostManList) if condition else 0
            scareOnDistList = [newScaredTimes[i] / item for (i, item) in enumerate(ghostManList)]
            condition = len(newManList) > 0
            pvar = pvariance(newManList) if condition else 0
            mn = mean(newManList) if condition else 0
            score = 5 * goodWay + 7 * mean(scareOnDistList) + ghostMeanDist - sqrt(ghostPvar)\
                -sqrt(pvar) - mn - indexPlusOne - uniform(0, 1)
            return score

        else:
            return -maxsize - 1 # we shouldn't die -> return minimum score

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        agentTypes = {0:NodeType.MAXIMIZER}
        addedDepths = 1
        while addedDepths < gameState.getNumAgents():
            addedDepths += 1
            agentTypes[addedDepths - 1] = NodeType.MINIMIZER

        tree = Tree(gameState, self.evaluationFunction, agentTypes, self.depth + 1, prune=False)
        _ = tree.find()
        return tree.getChosenAction()


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        agentTypes = {0:NodeType.MAXIMIZER}
        addedDepths = 1
        while addedDepths < gameState.getNumAgents():
            addedDepths += 1
            agentTypes[addedDepths - 1] = NodeType.MINIMIZER

        tree = Tree(gameState, self.evaluationFunction, agentTypes, self.depth + 1, prune=True)
        chosenValue = tree.find()
        chose = tree.getChosenAction()
        # print("best was:", chose, "with value:", chosenValue)
        # sleep(0.5)
        return chose

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        agentTypes = {0:NodeType.MAXIMIZER}
        addedDepths = 1
        while addedDepths < gameState.getNumAgents():
            addedDepths += 1
            agentTypes[addedDepths - 1] = NodeType.CHANCE_NODE

        tree = Tree(gameState, self.evaluationFunction, agentTypes, self.depth + 1, prune=False)
        _ = tree.find()
        return tree.getChosenAction()
                #         "--pacman=ExpectimaxAgent",
                # "--layout=smallClassic",
                # "--agentArgs=depth=2",


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    Don't forget to use pacmanPosition, foods, scaredTimers, ghostPositions!
    DESCRIPTION: <write something here so we know what you did>
    """

    pacmanPosition = currentGameState.getPacmanPosition()
    foods = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimers = [ghostState.scaredTimer for ghostState in ghostStates]
    ghostPositions = currentGameState.getGhostPositions()
    
    # "*** YOUR CODE HERE ***"

    # adding score of food
    score = currentGameState.getScore()
    foodScoreList = [2 ** -dist for dist in getManhattanList(pacmanPosition, foods.asList())]
    score += max(foodScoreList) if len(foodScoreList) > 0 else 0

    # adding score of being far away or eating ghosts
    ghostsManList = getManhattanList(pacmanPosition, ghostPositions)
    for i, dist in enumerate(ghostsManList):
        if scaredTimers[i] > 0:
            score += max(8 - dist, 0) ** 2
        else:
            score -= max(7 - dist, 0) ** 2

    # adding score of eating capsules
    capsulesManList = getManhattanList(pacmanPosition, currentGameState.getCapsules())
    capsulesScoreList = [50.0 / dist for dist in capsulesManList]
    score += max(capsulesScoreList) if len(capsulesScoreList) > 0 else 0

    return score

# Abbreviation
better = betterEvaluationFunction

def getManhattanList(point, ls):
    man_ls = []
    for pos in ls:
        man_ls.append(manhattanDistance(point, pos))
    
    return man_ls

def checkPrune(nodeType, v, alpha, beta): 
    if nodeType == NodeType.MAXIMIZER:
        if v > beta:
            return True
        return False
    elif nodeType == NodeType.MINIMIZER:
        if v < alpha:
            return True
        return False
        
class NodeType(Enum):
    MAXIMIZER = 'MAXIMIZER'
    MINIMIZER = 'MINIMIZER'
    CHANCE_NODE = 'CHANCE_NODE'


# an abstract class
class Node:
    def __init__(self, myRootState, evalFunc=scoreEvaluationFunction, value=None, id=0, agentTypes=None,\
         depth = 2, prune=False, alpha=None, beta=None):
        self.myRootState = myRootState
        self.depth = depth
        self.value = value
        self.prune = prune
        self.children = []
        self.evalFunc = evalFunc
        self.alpha = alpha
        self.beta = beta
        self.valueSet = False
        self.chosenAction = None
        self.id = id
        self.agentTypes = agentTypes
        
    def getValue(self):
        if self.valueSet:
            return self.value
            
        # the terminal (or pseudo terminal) nodes
        if (self.id == 0 and self.depth == 0) or self.myRootState.isWin() or self.myRootState.isLose(): 
            self.value = self.evalFunc(self.myRootState)
            self.valueSet = True
        else:
            legalActions = self.myRootState.getLegalActions(self.id)
            for action in legalActions:
                successorState = self.myRootState.generateSuccessor(self.id, action)
                givenDepth = self.depth - 1 if self.id == self.myRootState.getNumAgents() - 1 else self.depth
                newNodeID = (self.id + 1) % self.myRootState.getNumAgents()
                child = Tree.makeNode(successorState, self.evalFunc, newNodeID, self.agentTypes, givenDepth, self.prune,\
                     self.alpha, self.beta)
                self.children.append(child)
                changed = self.updateValue(child.getValue())
                # updating chosen action
                if changed:
                    self.chosenAction = action
                    self.valueSet = True
                if self.prune:
                    if checkPrune(self.agentTypes[self.id], self.value, self.alpha, self.beta):
                        return self.value
                    else:
                        self.updateAlphaBeta()

        return self.value
                    
    def updateValue(self, newValue):
        pass # abstract method

    def updateAlphaBeta(self):
        pass # abstract method

class Minimizer(Node):
    def __init__(self, myRootState, evalFunc=scoreEvaluationFunction, value=None, id=0, agentTypes=None,\
         depth=2, prune=False, alpha=None, beta=None):
        super().__init__(myRootState, evalFunc, value, id, agentTypes, depth, prune, alpha, beta)

    def updateValue(self, newValue):
        valCopy = self.value
        self.value = min(self.value, newValue)
        if valCopy != self.value:
            return True
        return False

    def updateAlphaBeta(self):
        self.beta = min(self.beta, self.value)

class Maximizer(Node):
    def __init__(self, myRootState, evalFunc=scoreEvaluationFunction, value=None, id=0, agentTypes=None,\
         depth=2, prune=False, alpha=None, beta=None):
        super().__init__(myRootState, evalFunc, value, id, agentTypes, depth, prune, alpha, beta)

    def updateValue(self, newValue):
        valCopy = self.value
        self.value = max(self.value, newValue)
        if valCopy != self.value:
            return True
        return False

    def updateAlphaBeta(self):
        self.alpha = max(self.alpha, self.value)

class ChanceNode(Node):
    def __init__(self, myRootState, evalFunc=scoreEvaluationFunction, value=None, id=0, agentTypes=None,\
         depth=2, prune=False, alpha=None, beta=None):
        super().__init__(myRootState, evalFunc, value, id, agentTypes, depth, prune, alpha, beta)

    def updateValue(self, newValue):
        valCopy = self.value
        actionsNum = len(self.myRootState.getLegalActions(self.id))
        if actionsNum == 0:
            return False
        
        self.value += newValue / actionsNum
        if valCopy != self.value:
            return True
        return False
    
class Tree:
    def __init__(self, rootState, evalFunc=scoreEvaluationFunction, agentTypes = {0:NodeType.MAXIMIZER},\
         depth = 2, prune=False):
        self.rootState = rootState
        self.agentTypes = agentTypes
        self.depth = depth
        self.prune = prune

        if len(self.agentTypes) <= 0:
            print("Invalid agentTypes length.")
            exit(-1)
        else:
            alpha = None
            beta = None
            if prune:
                from sys import maxsize
                alpha = -maxsize - 1
                beta = maxsize

            self.root = Tree.makeNode(rootState, evalFunc, 0, agentTypes, self.depth - 1, self.prune, alpha, beta)

    def find(self):
        return self.root.getValue()

    def getChosenAction(self):
        return self.root.chosenAction

    @classmethod
    def makeNode(cls, state, evalFunc, newNodeID, agentTypes, depth, prune, alpha, beta):
        from sys import maxsize
        node = None
        type = agentTypes[newNodeID]
        if type == NodeType.MAXIMIZER:
            node = Maximizer(state, evalFunc, -maxsize -1, newNodeID, agentTypes, depth, prune, alpha, beta)
        elif type == NodeType.MINIMIZER:
            node = Minimizer(state, evalFunc, maxsize, newNodeID, agentTypes, depth, prune, alpha, beta)
        elif type == NodeType.CHANCE_NODE:
            node = ChanceNode(state, evalFunc, 0, newNodeID, agentTypes, depth, prune, alpha, beta)
        else:
            print("Invalid NodeType is entered.")
            exit(-1)

        return node
