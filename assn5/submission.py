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
  def __init__(self):
    self.lastPositions = []
    self.dc = None


  def getAction(self, gameState):
    """
    getAction chooses among the best options according to the evaluation function.

    getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
    ------------------------------------------------------------------------------
    Description of GameState and helper functions:

    A GameState specifies the full game state, including the food, capsules,
    agent configurations and score changes. In this function, the |gameState| argument 
    is an object of GameState class. Following are a few of the helper methods that you 
    can use to query a GameState object to gather information about the present state 
    of Pac-Man, the ghosts and the maze.
    
    gameState.getLegalActions(): 
        Returns the legal actions for the agent specified. Returns Pac-Man's legal moves by default.

    gameState.generateSuccessor(agentIndex, action): 
        Returns the successor state after the specified agent takes the action. 
        Pac-Man is always agent 0.

    gameState.getPacmanState():
        Returns an AgentState object for pacman (in game.py)
        state.configuration.pos gives the current position
        state.direction gives the travel vector

    gameState.getGhostStates():
        Returns list of AgentState objects for the ghosts

    gameState.getNumAgents():
        Returns the total number of agents in the game

    gameState.getScore():
        Returns the score corresponding to the current state of the game
        It corresponds to Utility(s)

    
    The GameState class is defined in pacman.py and you might want to look into that for 
    other helper methods, though you don't need to.
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best


    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

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

######################################################################################
# Problem 1a: implementing minimax

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (problem 1)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction. Terminal states can be found by one of the following: 
      pacman won, pacman lost or there are no legal moves. 

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game

      gameState.getScore():
        Returns the score corresponding to the current state of the game
        It corresponds to Utility(s)
    
      gameState.isWin():
        Returns True if it's a winning state
    
      gameState.isLose():
        Returns True if it's a losing state

      self.depth:
        The depth to which search should continue
    """

    # BEGIN_YOUR_ANSWER (our solution is 30 lines of code, but don't worry if you deviate from this)
    def V(state, agent, depth):
      actions = state.getLegalActions(agent)

      if state.isWin() or state.isLose() or actions == [Directions.STOP]:
        return state.getScore(), Directions.STOP
      
      if depth == 0:
        return self.evaluationFunction(state), Directions.STOP
      
      Vs = [(V(state.generateSuccessor(agent, action),
               agent + 1 if agent < state.getNumAgents() - 1 else 0,
               depth if agent < state.getNumAgents() - 1 else depth - 1)[0], action)
            for action in actions if action != Directions.STOP]
      
      if agent == self.index:
        return max(Vs)
      else:
        return min(Vs)

    return V(gameState, self.index, self.depth)[1]
    # END_YOUR_ANSWER

######################################################################################
# Problem 2a: implementing alpha-beta

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (problem 2)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """

    # BEGIN_YOUR_ANSWER (our solution is 42 lines of code, but don't worry if you deviate from this)
    def V(state, agent, depth, alpha, beta):
      actions = state.getLegalActions(agent)

      if state.isWin() or state.isLose() or actions == [Directions.STOP]:
        return state.getScore(), Directions.STOP
      
      if depth == 0:
        return self.evaluationFunction(state), Directions.STOP
      
      if agent < state.getNumAgents() - 1:
        newAgent = agent + 1
        newDepth = depth
      else:
        newAgent = 0
        newDepth = depth - 1

      if agent == self.index:
        maxV = float("-inf"), Directions.STOP
        for action in actions:
          if action == Directions.STOP: continue
          maxV = max(maxV, (V(state.generatePacmanSuccessor(action), newAgent, newDepth, alpha, beta)[0], action))
          alpha = max(alpha, maxV[0])
          if beta < alpha: break
        return maxV
      else:
        minV = float("inf"), Directions.STOP
        for action in actions:
          if action == Directions.STOP: continue
          minV = min(minV, (V(state.generateSuccessor(agent, action), newAgent, newDepth, alpha, beta)[0], action))
          beta = min(beta, minV[0])
          if beta < alpha: break
        return minV

    return V(gameState, self.index, self.depth, float("-inf"), float("inf"))[1]
    # END_YOUR_ANSWER

######################################################################################
# Problem 3a: implementing expectimax

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (problem 3)
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """

    # BEGIN_YOUR_ANSWER (our solution is 30 lines of code, but don't worry if you deviate from this)
    def V(state, agent, depth):
      actions = state.getLegalActions(agent)

      if state.isWin() or state.isLose() or actions == [Directions.STOP]:
        return state.getScore(), Directions.STOP
      
      if depth == 0:
        return self.evaluationFunction(state), Directions.STOP
      
      if agent < state.getNumAgents() - 1:
        newAgent = agent + 1
        newDepth = depth
      else:
        newAgent = 0
        newDepth = depth - 1

      if agent == self.index:
        return max((V(state.generateSuccessor(agent, action), newAgent, newDepth)[0], action)
                   for action in actions if action != Directions.STOP)
      else:
        return sum(V(state.generateSuccessor(agent, action), newAgent, newDepth)[0] / len(actions) for action in actions), None

    return V(gameState, self.index, self.depth)[1]
    # END_YOUR_ANSWER

######################################################################################
# Problem 4a (extra credit): creating a better evaluation function

def betterEvaluationFunction(currentGameState):
  """
  Your extreme, unstoppable evaluation function (problem 4).
  """

  # BEGIN_YOUR_ANSWER (our solution is 60 lines of code, but don't worry if you deviate from this)
  currentPacmanPosition = currentGameState.getPacmanPosition()
  features, weights = [], []

  # default score
  features.append(currentGameState.getScore())
  weights.append(1)

  # manhattan distance to the closest ghost
  distToGhosts, distToScaredGhosts = [], []
  for ghost in currentGameState.getGhostStates():
    if ghost.scaredTimer == 0:
      distToGhosts.append(manhattanDistance(currentPacmanPosition, ghost.getPosition()))
    else:
      distToScaredGhosts.append(ghost.scaredTimer / manhattanDistance(currentPacmanPosition, ghost.getPosition()))
  features.append(min(distToGhosts) if len(distToGhosts) > 0 else 0)
  weights.append(1)
  features.append(min(distToScaredGhosts) if len(distToScaredGhosts) > 0 else 0)
  weights.append(15)

  # number of foods
  numFood = currentGameState.getNumFood()
  features.append(numFood)
  weights.append(-0.1)

  # manhattan distance to the closest food
  distToFoods = [manhattanDistance(currentPacmanPosition, foodPosition) for foodPosition in currentGameState.getFood().asList()]
  features.append((1 / min(distToFoods)) if len(distToFoods) > 0 else 0)
  weights.append(1 / (numFood ** 2))

  # manhattan distance to the closest capsule
  distToCapsules = [1 / manhattanDistance(currentPacmanPosition, capsulePosition) for capsulePosition in currentGameState.getCapsules()]
  features.append(min(distToCapsules) if len(distToCapsules) > 0 else 0)
  weights.append(5 if len(distToScaredGhosts) == 0 else 0)

  # score
  return sum(feature * weight for feature, weight in zip(features, weights))
  # END_YOUR_ANSWER

# Abbreviation
better = betterEvaluationFunction

