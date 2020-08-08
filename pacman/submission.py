from util import manhattanDistance
from game import Directions
import random, util
import math
from game import Agent
from operator import itemgetter

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
    
    # print(legalMoves[chosenIndex])
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
  
  # currentGameState.data.score += currentGameState.data.scoreChange  
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
# Problem 1b: implementing minimax

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (problem 1)
  """

  def Minimax(self,gameState,currentDepth,AgentNumber):
    if currentDepth==1 or len(gameState.getLegalActions(AgentNumber))==0:
      return (scoreEvaluationFunction(gameState),None)
    if AgentNumber==0:
      AgentNumber=gameState.getNumAgents()-1
      actions=gameState.getLegalActions(0)
      maximum=-math.inf
      bestChoices=[]
      for action in actions:
        value=self.Minimax(gameState.generatePacmanSuccessor(action),currentDepth-1,AgentNumber)[0]
        if value>maximum:
          maximum=value
          bestChoices=[action]
        elif value==maximum:
          bestChoices.append(action)
      return maximum,random.choice(bestChoices)
    else:
      actions=gameState.getLegalActions(AgentNumber)
      minimum=math.inf
      bestChoices=[]
      for action in actions:
        value=self.Minimax(gameState.generateSuccessor(AgentNumber,action),currentDepth-1,AgentNumber-1)[0]
        if value<minimum:
          minimum=value
          bestChoices=[action]
        elif value==minimum:
          bestChoices.append(action)
      return minimum,random.choice(bestChoices)

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction. Terminal states can be found by one of the following:
      pacman won, pacman lost or there are no legal moves.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game

      gameState.getScore():
        Returns the score corresponding to the current state of the game

      gameState.isWin():
        Returns True if it's a winning state

      gameState.isLose():
        Returns True if it's a losing state

      self.depth:
        The depth to which search should continue

    """

    # BEGIN_YOUR_CODE (our solution is 20 lines of code, but don't worry if you deviate from this)
    return (self.Minimax(gameState,self.depth*( gameState.getNumAgents() )+1,0)[1])
    # raise Exception("Not implemented yet")
    # END_YOUR_CODE

######################################################################################
# Problem 2a: implementing alpha-beta

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (problem 2)
  """
  def getCount(self,tuple):
    if type(tuple[0]) is int or type(tuple[0]) is float:
      return tuple[0]
    else:
      return self.getCount(tuple[0])
  
  def MinimaxWithPruning(self,gameState,currentDepth,AgentNumber,alpha,beta):
    if currentDepth==1 or len(gameState.getLegalActions(AgentNumber))==0:
      return (scoreEvaluationFunction(gameState),Directions.STOP)
    
    #Max node ie. pacman
    if AgentNumber==0:
      bestActions=[]
      AgentNumber=gameState.getNumAgents()-1
      actions=gameState.getLegalActions(0)
      bestVal=-math.inf
      for action in actions:
        value=(self.MinimaxWithPruning(gameState.generatePacmanSuccessor(action),currentDepth-1,AgentNumber,alpha,beta))[0]
        if bestVal<value:
          bestActions=[action]
          bestVal=value
        elif bestVal==value:
          bestActions.append(action)
        alpha=max(alpha,value)
        if alpha>=beta:
          break
      return bestVal,random.choice(bestActions)
      
    else:
      actions=gameState.getLegalActions(AgentNumber)
      bestActions=[]
      bestVal=math.inf
      for action in actions:
        value=self.MinimaxWithPruning(gameState.generateSuccessor(AgentNumber,action),currentDepth-1,AgentNumber-1,alpha,beta)[0]
        if bestVal>value:
          bestVal=value
          bestActions=[action]
        elif bestVal==value:
          bestActions.append(action)
        beta=min(beta,value)
        if alpha>=beta:
          break
      return bestVal,random.choice(bestActions)
      
  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """

    # BEGIN_YOUR_CODE (our solution is 36 lines of code, but don't worry if you deviate from this)
    response=self.MinimaxWithPruning(gameState,self.depth*( gameState.getNumAgents() )+1,0,-math.inf,math.inf)
    return response[1]

    # END_YOUR_CODE

######################################################################################
# Problem 3b: implementing expectimax

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (problem 3)
  """
  def betterEvaluationFunction(self,currentGameState):
    """
      Your extreme, unstoppable evaluation function (problem 4). Note that you can't fix a seed in this function.

      DESCRIPTION: <write something here so we know what you did>
    """
    
    # BEGIN_YOUR_CODE (our solution is 13 lines of code, but don't worry if you deviate from this)
    #using information from current game state and successor game state
    # successorGameState = currentGameState.generatePacmanSuccessor(action)
    # newPos = successorGameState.getPacmanPosition()
    Food = currentGameState.getFood()
    FoodList=Food.asList()
    CapsuleList = currentGameState.getCapsules()
    # print(Capsules)
    # CapsuleList=Capsules.asList()
    GhostStates = currentGameState.getGhostStates()
    PacManStates=currentGameState.getPacmanState()
    # GhostPositions=[]
    GhostPositions=currentGameState.getGhostPositions()
    # print(GhostPositions)
    PacmanPosition=currentGameState.getPacmanPosition()
    newScaredTimes = [ghostState.scaredTimer for ghostState in GhostStates]

    if currentGameState.isLose():
      currentGameState.data.scoreChange -=100
    # if currentGameState.isWin():
    #   return math.inf
    
    GhostDist=[]
    for i in range(currentGameState.getNumAgents()-1):  #1. it checks if ghost is nearby and give a fucking big negative value if it is very nearby
      GhostDist.append(abs(GhostPositions[i][0]-PacmanPosition[0])+abs(GhostPositions[i][1]-PacmanPosition[1]))
      if 0 in GhostDist:
        currentGameState.data.scoreChange -=100
    if sum(GhostDist)<6:
      currentGameState.data.scoreChange -=50/(sum(GhostDist))
      
    #now comes the magic
    
    currentGameState.data.scoreChange +=120/(len(FoodList)+1)
    FoodDist=[]
    for oneFood in FoodList:
      FoodDist.append(abs(oneFood[0]-PacmanPosition[0])+abs(oneFood[1]-PacmanPosition[1]))
    
    CapsuleDist=[]
    for oneCapsule in CapsuleList:
      CapsuleDist.append(abs(oneCapsule[0]-PacmanPosition[0])+abs(oneCapsule[1]-PacmanPosition[1]))

    currentGameState.data.scoreChange -= 10*sum(FoodDist)+5*sum(CapsuleDist)
    
    currentGameState.data.score += currentGameState.data.scoreChange
    return currentGameState.getScore()
    # END_YOUR_CODE

  def Expectimax(self,gameState,currentDepth,AgentNumber):
    if currentDepth==1 or len(gameState.getLegalActions(AgentNumber))==0:
      return (self.betterEvaluationFunction(gameState),Directions.STOP)
    if AgentNumber==0:
      AgentNumber=gameState.getNumAgents()-1
      actions=gameState.getLegalActions(0)
      maximum=-math.inf
      bestChoices=[]
      for action in actions:
        value=self.Expectimax(gameState.generatePacmanSuccessor(action),currentDepth-1,AgentNumber)[0]
        if value>maximum:
          maximum=value
          bestChoices=[action]
        elif value==maximum:
          bestChoices.append(action)
      return maximum,random.choice(bestChoices)
    else:
      actions=gameState.getLegalActions(AgentNumber)
      values=[]
      for action in actions:
        values.append((self.Expectimax(gameState.generateSuccessor(AgentNumber,action),currentDepth-1,AgentNumber-1)[0]))
      
      return sum(values)/len(values),random.choice(actions)


  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """
    # BEGIN_YOUR_CODE (our solution is 20 lines of code, but don't worry if you deviate from this)
    return (self.Expectimax(gameState,self.depth*(gameState.getNumAgents())+1,0)[1])
    # raise Exception("Not implemented yet")
    # END_YOUR_CODE

######################################################################################
# Problem 4a (extra credit): creating a better evaluation function


# Abbreviation
# better = betterEvaluationFunction
