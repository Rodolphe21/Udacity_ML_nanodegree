import random
import math
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """ An agent that learns to drive in the Smartcab world.
        This is the object you will be modifying. """ 

    def __init__(self, env, learning=False, epsilon=1.0, alpha=0.5):
        super(LearningAgent, self).__init__(env)     # Set the agent in the evironment 
        self.planner = RoutePlanner(self.env, self)  # Create a route planner
        self.valid_actions = self.env.valid_actions  # The set of valid actions

        # Set parameters of the learning agent
        self.learning = learning # Whether the agent is expected to learn
        self.Q = dict()          # Create a Q-table which will be a dictionary of tuples
        self.epsilon = epsilon   # Random exploration factor
        self.alpha = alpha       # Learning factor

        # Set any additional class parameters as needed
        self.trial_counter = 0 # integer value to count trials and input for the decay function, initially 0
        
    def reset(self, destination=None, testing=False):
        """ The reset function is called at the beginning of each trial.
            'testing' is set to True if testing trials are being used
            once training trials have completed. """

        # Select the destination as the new location to route to
        self.planner.route_to(destination)
        
        # Update epsilon using a decay function of your choice
        # Update additional class parameters as needed
        # If 'testing' is True, set epsilon and alpha to 0
        
        print "\n +++ reset() +++ epsilon is: ", self.epsilon
        print "\n +++ reset() +++ trial_counter is: ", self.trial_counter
        # print "\n +++ reset() +++ alpha: ", self.alpha
        
        if (testing == True): 
            self.epsilon = 0
            self.alpha = 0
        
        if ( (self.learning == True) and (testing == False) ): 
            self.epsilon = math.exp(-self.trial_counter*0.01)      # use 'self.trial_counter' later on when the decay function is not linear any longer
   
        self.trial_counter = self.trial_counter + 1 # we increment self.trial_counter each time we run reset(), each time we start a new trial

### BETTER CODING ### self.trial_counter += 1
   
### OTHER RECOMMENDATION FOR THE DECAY FUNCTION : A SIGMOID ###
# self.trial_count += 1
# self.epsilon = 1 - (1/(1+math.exp(-k*self.alpha*(self.trial_count-t0))))
#    Where k determines how fast the agent performs the transition between random learning and choosing the max q-value. 
#    k also determines how fast the sigmoid function converges to 0.
#    The t0 value can be chosen empirically; by using the sigmoid function, we can make sure that the agent would have 
#    sufficient time to explore the environment completely randomly, in order also to fill the Q-value matrix with the correct values.
###

        return None

    def build_state(self):
        """ The build_state function is called when the agent requests data from the 
            environment. The next waypoint, the intersection inputs, and the deadline 
            are all features available to the agent. """

        # Collect data about the environment
        waypoint = self.planner.next_waypoint() # The next waypoint 
        inputs = self.env.sense(self)           # Visual input - intersection light and traffic
        deadline = self.env.get_deadline(self)  # Remaining deadline

        # NOTE : you are not allowed to engineer features outside of the inputs available.
        # Because the aim of this project is to teach Reinforcement Learning, we have placed 
        # constraints in order for you to learn how to adjust epsilon and alpha, and thus learn about the balance between exploration and exploitation.
        # With the hand-engineered features, this learning process gets entirely negated.
        
        # Set 'state' as a tuple of relevant data for the agent        
        # print "visual inputs are:", inputs
        current_light = inputs['light']
        # print "\ncurrent_light is: ", current_light
        current_oncoming = inputs['oncoming']
        current_right = inputs['right']
        current_left = inputs['left']
        next_move = waypoint
        state = (current_light, current_oncoming, current_right, current_left, next_move)
        # print "\n +++ build_state() +++ state is: ", state
        
        return state


    def get_maxQ(self, state):
        """ The get_max_Q function is called when the agent is asked to find the
            maximum Q-value of all actions based on the 'state' the smartcab is in. """

        # Calculate the maximum Q-value of all actions for a given state
        maxQ = max(self.Q[state][None], self.Q[state]['forward'], self.Q[state]['left'], self.Q[state]['right'])
        # print "\n +++ get_maxQ() +++ maxQ is: ", maxQ
        
### BETTER CODING ### maxQ = max(self.Q[state].values())
        
        return maxQ
    

    def createQ(self, state):
        """ The createQ function is called when a state is generated by the agent. """

        # When learning, check if the 'state' is not in the Q-table
        # If it is not, create a new dictionary for that state
        #   Then, for each action available, set the initial Q-value to 0.0
        if (self.learning == True) and state not in self.Q: 
            self.Q[state] = {None: 0, 'forward': 0, 'left':0, 'right': 0} # these are the Q values for the respective possible actions
            # print "\n +++ createQ() +++ learning +++ state NOT in Q, now initialised"
        # else: print "\n +++ createQ() +++ learning +++ state in Q"

### BETTER CODING ### state = (waypoint, inputs['light'], inputs['left'], inputs['oncoming'], inputs['right'])
        
        return


    def choose_action(self, state):
        """ The choose_action function is called when the agent is asked to choose
            which action to take, based on the 'state' the smartcab is in. """

        # Set the agent state and default action
        self.state = state
        self.next_waypoint = self.planner.next_waypoint()
        
        # When not learning, choose a random action
        if (self.learning == False): action = random.choice(self.valid_actions)
      
        # When learning, choose a random action with 'epsilon' probability
        # Otherwise, choose an action with the highest Q-value for the current state
        # Be sure that when choosing an action with highest Q-value that you randomly select between actions that "tie".
        
        # print "\n +++ valid actions are: ", self.valid_actions
        # print "\n +++ choose_action() +++ random check +++: ", random.random()
        
        list_of_actions_with_max_Qval = []
        
        if (self.learning == True):
            if (random.random() < self.epsilon): # with epsilon decaying, the probability of random actions decreases (less exploration)
                action = random.choice(self.valid_actions)
                # print "\n +++ choose_action() +++ learning +++ RANDOM action: ", action
            else:
                # print "\n +++ choose_action() +++ qVal None: ", self.Q[state][None] 
                # print "+++ choose_action() +++ qVal 'forward': ", self.Q[state]['forward']
                # print "+++ choose_action() +++ qVal 'left': ", self.Q[state]['left']
                # print "+++ choose_action() +++ qVal 'right': ", self.Q[state]['right']
                maxQ = self.get_maxQ(state)
                # print "+++ choose_action() +++ get_maxQ() says the highest Qval is: ", maxQ
            
            
                for action in self.Q[state]:
                    if (self.Q[state][action] == maxQ): list_of_actions_with_max_Qval.append(action)
                
                # print "\n +++ choose_action() +++ list_of_actions_with_max_Qval: ", list_of_actions_with_max_Qval
                action = random.choice(list_of_actions_with_max_Qval)
                # print "\n +++ choose_action() +++ learning +++ action w/ HIGHEST Qval: ", action

### BETTER CODING ### best_actions = [action for action in self.valid_actions if self.Q[state][action] == self.get_maxQ(state)]
### BETTER CODING ### action = random.choice(best_actions)

        return action


    def learn(self, state, action, reward):
        """ The learn function is called after the agent completes an action and
            receives a reward. This function does not consider future rewards 
            when conducting learning. """

        ########### 
        ## TO DO ##
        ###########
        # When learning, implement the value iteration update rule
        #   Use only the learning rate 'alpha' (do not use the discount factor 'gamma')

        if ( self.learning == True): 
            # print "\n +++ learn() +++ state: ", state
            # print "\n +++ learn() +++ action: ", action
            # print "\n +++ learn() +++ reward: ", reward
            # print "\n +++ learn() +++ self.Q[state][action]: ", self.Q[state][action]
            # print "\n +++ learn() +++ self.alpha: ", self.alpha
            
            self.Q[state][action] = ( (1-self.alpha)*self.Q[state][action] + self.alpha*reward ) # approximated Q-Learning algo w/ gamma = 0
            # print "\n +++ learn() +++ new Qval: ", self.Q[state][action]
        
        return


    def update(self):
        """ The update function is called when a time step is completed in the 
            environment for a given trial. This function will build the agent
            state, choose an action, receive a reward, and learn if enabled. """

        state = self.build_state()          # Get current state
        self.createQ(state)                 # Create 'state' in Q-table
        action = self.choose_action(state)  # Choose an action
        reward = self.env.act(self, action) # Receive a reward
        self.learn(state, action, reward)   # Q-learn
        
        return
        

def run():
    """ Driving function for running the simulation. 
        Press ESC to close the simulation, or [SPACE] to pause the simulation. """

    ##############
    # Create the environment
    # Flags:
    #   verbose     - set to True to display additional output from the simulation
    #   num_dummies - discrete number of dummy agents in the environment, default is 100
    #   grid_size   - discrete number of intersections (columns, rows), default is (8, 6)
    env = Environment(verbose=True)
    
    ##############
    # Create the driving agent
    # Flags:
    #   learning   - set to True to force the driving agent to use Q-learning
    #    * epsilon - continuous value for the exploration factor, default is 1
    #    * alpha   - continuous value for the learning rate, default is 0.5
    agent = env.create_agent(LearningAgent, learning=True)
    
    ##############
    # Follow the driving agent
    # Flags:
    #   enforce_deadline - set to True to enforce a deadline metric
    env.set_primary_agent(agent,enforce_deadline=True)

    ##############
    # Create the simulation
    # Flags:
    #   update_delay - continuous time (in seconds) between actions, default is 2.0 seconds
    #   display      - set to False to disable the GUI if PyGame is enabled
    #   log_metrics  - set to True to log trial and simulation results to /logs
    #   optimized    - set to True to change the default log file name
    sim = Simulator(env, update_delay=0.01,log_metrics=True, optimized=True)
    
    ##############
    # Run the simulator
    # Flags:
    #   tolerance  - epsilon tolerance before beginning testing, default is 0.05 
    #   n_test     - discrete number of testing trials to perform, default is 0
    sim.run(n_test=10)


if __name__ == '__main__':
    run()
