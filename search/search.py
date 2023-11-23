# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    # Verifica se lo stato iniziale è il goal
    statstart = problem.getStartState()
    if problem.isGoalState(statstart):
        return []
    # Inizializza uno stack per la ricerca in profondità (DFS) e un insieme per tenere traccia degli stati visitati
    coda = util.Stack()
    visto = set()
    # Inserisce nello stack lo stato iniziale e un percorso vuoto
    coda.push((statstart, []))
    #controllo che lo stack non sia vuoto nel mentre
    while not coda.isEmpty():
        # Estrae dallo stack uno stato e il percorso per raggiungerlo
        current_state, path = coda.pop()
        # Salta l'elaborazione se lo stato è già stato visitato
        if current_state in visto:
            continue

    # Aggiunge lo stato corrente all'insieme degli stati visitati
        visto.add(current_state)
    # Verifica se lo stato corrente è lo stato obiettivo
        if problem.isGoalState(current_state):
        # Restituisce il percorso che porta al goal
            return path  
    # Esplora i successori dello stato corrente
        for successor, action, _ in problem.getSuccessors(current_state):
            if successor not in visto:
            # Inserisce nello stack il successore e il percorso aggiornato
                coda.push((successor, path + [action]))
    # Restituisce una lista vuota se non viene trovato
    # alcun percorso verso lo stato obiettivo
    return []
    util.raiseNotDefined()

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # Verifica se lo stato iniziale corrisponde già al Goal
    statstart = problem.getStartState()
    if problem.isGoalState(statstart):
        # Se lo stato iniziale è lo stato obiettivo, 
        # non è necessario eseguire altre azioni.
        # Restituisce una lista vuota, 
        # indicando che non sono richiesti ulteriori passaggi.
        return []
    # Inizializza una coda per la ricerca in ampiezza (BFS) e un insieme per tenere traccia degli stati visitati
    visto = set([statstart])  # Marca lo stato iniziale come visitato
    coda = util.Queue()
    # Inserisce nella coda lo stato iniziale con un percorso vuoto
    coda.push((statstart, []))
    #controllo che la queue non sia vuota nel mentre
    while not coda.isEmpty():
        # Estrae dalla coda uno stato e il percorso per raggiungerlo
        current_state, path = coda.pop()
        # Verifica se lo stato corrente è il goal
        if problem.isGoalState(current_state):
        # Se lo stato corrente è il goal, 
        # restituisce il percorso per raggiungerlo.
            return path  
        # Esplora i successori dello stato corrente
        for successor, action, _ in problem.getSuccessors(current_state):
            if successor not in visto:
        # Se il successore non è stato visitato, 
        # lo aggiunge all'insieme degli stati visitati
                visto.add(successor)  
        # Aggiunge l'azione al percorso corrente e 
        # inserisce nella coda il successore con il nuovo percorso
                coda.push((successor, path + [action]))
        # Restituisce una lista vuota se non viene trovato alcun percorso verso il goal
        # Questo si verifica quando tutti gli stati possibili sono stati visitati 
        # senza raggiungere il goal
    return []
    util.raiseNotDefined()

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    # Crea una coda con priorità per gestire i nodi da esplorare
    codapriority = util.PriorityQueue()
    statstart = problem.getStartState()
    # Inserisce lo stato iniziale nella coda con priorità,
    #con un percorso e costo iniziali
    codapriority.push((statstart, [], 0), 0) 
                    # ((stato, percorso, costo), priorità)
    visited = {}
    # Dizionario per tenere traccia del costo più basso per raggiungere
    # gli stati visitati
    while not codapriority.isEmpty():
        # Estrae il nodo con il costo totale più basso dalla coda con priorità
        current_state, path, current_cost = codapriority.pop()
        # Verifica se lo stato corrente è lo stato obiettivo
        if problem.isGoalState(current_state):
        # Se lo stato corrente è lo stato obiettivo, 
        #restituisce il percorso per raggiungerlo
            return path
        # Verifica se lo stato è stato visitato con un costo maggiore
        if current_state not in visited or current_cost < visited[current_state]:
        # Aggiorna il costo più basso per raggiungere lo stato corrente
            visited[current_state] = current_cost
        # Esplora tutti i successori dello stato corrente
            for successor, action, step_cost in problem.getSuccessors(current_state):
        # Calcola il nuovo costo totale per raggiungere il successore
                new_cost = current_cost + step_cost
        # Aggiorna il percorso con l'azione corrente
                new_path = path + [action]
        # Aggiorna la coda con priorità con il successore, 
        # il nuovo percorso e il nuovo costo
                codapriority.update((successor, new_path, new_cost), new_cost)
        # Restituisce una lista vuota se non viene trovato alcun percorso
        # verso lo stato obiettivo
    return []

    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    # Inizializza una coda con priorità inserendo lo stato iniziale
    statstart = problem.getStartState()
    codapriority = util.PriorityQueue()
    # Inserisce lo stato iniziale nella coda: (stato, percorso, costo), priorità
    codapriority.push((statstart, [], 0), 0)
                    #((stato, percorso, costo), priorità)
    visited = {}
    # Dizionario per tenere traccia del miglior costo per
    # raggiungere gli stati visitati
    while not codapriority.isEmpty():
        # Estrae dalla coda il nodo con la priorità più alta (costo più basso)
        current_state, path, current_cost = codapriority.pop()
        # Verifica se lo stato corrente è lo stato obiettivo
        if problem.isGoalState(current_state):
            # Se è lo stato obiettivo, restituisce il percorso per raggiungerlo
            return path
        # Controlla se lo stato è stato visitato con un costo più elevato
        if current_state not in visited or current_cost < visited[current_state]:
            # Aggiorna il costo migliore per raggiungere lo stato corrente
            visited[current_state] = current_cost
            # Esplora i successori dello stato corrente
            #itero
            for successor, action, step_cost in problem.getSuccessors(current_state):
                # Calcola il nuovo costo complessivo per raggiungere il successore
                new_cost = current_cost + step_cost
                # Calcola il costo stimato per raggiungere il Goal
                # partendo dal successore
                estimated_cost_to_goal = new_cost + heuristic(successor, problem)
                # Aggiorna il percorso includendo l'azione corrente
                new_path = path + [action]
                # Aggiorna la coda con priorità con il successore, 
                # il nuovo percorso e il nuovo costo stimato
                codapriority.update((successor, new_path, new_cost), estimated_cost_to_goal)
    # Restituisce una lista vuota se non viene trovato alcun percorso 
    # verso lo stato obiettivo
    return []

    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
