# Pacman-AI
A pacman game from an assignment of Stanford AI course CS221
A multi-agent pacman game which has muliple ghosts and we need to find the best strategy for pacman to win

 I implemeted:
 1. MINIMAX
 2. EXPECTIMAX
 3. ALPHA-BETA PRUNING
 to the agent AI.
 
 Also implemted an evaluation function to calculate the utilty of a state using variables like ghosts, their position, fruit position, pill position and time taken, etc.
 
commands:
  python pacman.py -p <Agent Strategy> -l <MapLayout> -k <No. of ghosts>
for example:
  python pacman.py -p AlphaBetaAgent -a depth=3 -k 3

