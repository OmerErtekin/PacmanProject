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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()
        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        state = gameState.generatePacmanSuccessor(legalMoves[chosenIndex])
        pacPos = state.getPacmanPosition()
        foodPositions = GetFoodPositions(state.getFood())

        if len(foodPositions) > 0:
            distances = GetFoodDistances(pacPos,foodPositions)
            print(distances)


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
        return successorGameState.getScore()

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
        #burada bi aksiyon dönmeli
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


class ExpectimaxAgent(MultiAgentSearchAgent):
    def ExpectimaxValue(self,gameState,agentIndex,nodeDepth):
        if agentIndex >= gameState.getNumAgents():
            agentIndex = 0
            nodeDepth +=1
        
        if nodeDepth == self.depth:
            return self.evaluationFunction(gameState)
        if agentIndex == self.index:
            return self.MaxValue(gameState,agentIndex,nodeDepth)
        else:
            return self.ExpectedValue(gameState,agentIndex,nodeDepth)


    def getAction(self, gameState):
        move = self.ExpectimaxValue(gameState,0,0)
        return move
        

    def MaxValue(self,currentGameState,agentIndex,nodeDepth):
        
        if currentGameState.isWin() or currentGameState.isLose():
            return self.evaluationFunction(currentGameState)
        
        currentMax = float("-inf")
        selectedAction = "Stop"

        for legalAction in currentGameState.getLegalActions(agentIndex):
            if legalAction == Directions.STOP:
                continue
            
            successor = currentGameState.generateSuccessor(agentIndex, legalAction)
            expectimaxValue = self.ExpectimaxValue(successor,agentIndex+1,nodeDepth)

            if expectimaxValue > currentMax:
                currentMax = expectimaxValue
                selectedAction = legalAction
            
        if nodeDepth == 0:
            return selectedAction
        else:
            return currentMax
            
                

    def ExpectedValue(self,currentGameState,agentIndex,nodeDepth):
        if currentGameState.isWin() or currentGameState.isLose():
            return self.evaluationFunction(currentGameState)
        
        expectedValue = 0
        probabilty = 1.0/len(currentGameState.getLegalActions(agentIndex))

        for legalAction in currentGameState.getLegalActions(agentIndex):
            if legalAction == Directions.STOP:
                continue

            successor = currentGameState.generateSuccessor(agentIndex, legalAction)            
            currentValue = self.ExpectimaxValue(successor,agentIndex+1,nodeDepth)
            expectedValue += currentValue * probabilty

        return expectedValue


def betterEvaluationFunction(currentGameState):

    pacmanPos = currentGameState.getPacmanPosition()
    currentGhostStates =  currentGameState.getGhostStates()
    currentCapsules = currentGameState.getCapsules()
    currentScore = currentGameState.getScore()

    foodList = GetFoodPositions(currentGameState.getFood())
    distances = GetFoodDistances(currentGameState.getPacmanPosition(),foodList)

    nearFoodPoint = 0
    if len(distances) > 0:
        nearFoodPoint = 0.5 / distances[0]['Distance']

        
    ghostDistance = float("-inf")
    scaredGhost = 0

    for ghostState in currentGhostStates:
        ghostPos = ghostState.getPosition()
        if pacmanPos == ghostPos:
            #if there is an ai and we can eat it, go for it, if there is an ghost that you can't eat, escape if you can
            if ghostState.scaredTimer <= 0:
                return float("-inf") 
            else:
                return float("inf")
        else:
            if ghostState.scaredTimer <= 0:
                ghostDistance = min(ghostDistance,manhattanDistance(pacmanPos,ghostPos))
            else:
                ghostDistance = max(ghostDistance,manhattanDistance(pacmanPos,ghostPos))

        if ghostState.scaredTimer != 0:
            scaredGhost += 1
    
    capsuleDistance = float("-inf")

    for capsuleState in currentCapsules:
        capsuleDistance = min(capsuleDistance,manhattanDistance(pacmanPos,capsuleState))

    ghostDistance = 1.0/ (1.0 + (ghostDistance/(len(currentGhostStates))))
    capsuleDistance = 2.0/(1.0 + (len(currentCapsules)))
    scaredGhost = 1.0/(1.0 + scaredGhost)

    return currentScore + nearFoodPoint + ghostDistance + capsuleDistance


def GetFoodPositions(state):
    foodIndexList = []
    for i in range (state.width):
        for j in range (state.height):
            if state[i][j] == True:
                foodIndexList.append((i,j))
    return foodIndexList
        
def GetFoodDistances(pacmanPos,foodPositions):
    distanceList = []
    for foodPos in foodPositions:
        distanceList.append( {'Distance' : manhattanDistance(pacmanPos,foodPos), 'FoodPos' : foodPos} )

    distanceList = sorted(distanceList,key = lambda x : x['Distance'])
    #example output : [ {'Distance': 2, 'FoodPos': (3, 1)}, {'Distance': 4, 'FoodPos': (1, 5)}, {'Distance': 4, 'FoodPos': (3, 3)}, {'Distance': 6, 'FoodPos': (1, 7)}, 
    # {'Distance': 6, 'FoodPos': (2, 6)}, {'Distance': 6, 'FoodPos': (3, 5)}, {'Distance': 8, 'FoodPos': (2, 8)}, {'Distance': 8, 'FoodPos': (3, 7)} ]
    return distanceList


        

# Abbreviation
better = betterEvaluationFunction
