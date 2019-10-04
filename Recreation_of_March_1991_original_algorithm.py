'''
March 1991

@author: Maciej Workiewicz (ESSEC)
date: September 25, 2019

This is a recreation of the computational model from March JG. 1991. Exploration
and Exploitation in Organizational Learning, Organization Science. 2(1):71-87

The code is based on the description from the paper itself (the relevant
quotes are included in the code with a reference to page on which they appear
in the original manuscript), but also on Rodan (2005) and especially the note
by Michael Christensen (2015), both of which describe the missing steps
necessary to recreate Figure 1 from the original paper. Attempting recreation
of the model from the 1991 paper alone is not possible as the original
text misses some important information. I provided specific notes in those
places where the original code differs from the paper.

The objective of this exercise was to reproduce Figure 1, but the code
can also be used to reproduce additional results from the paper, like
impact of employee turnover or environmental turbulence.

The code has been tested on Anaconda Python 3.7
'''

import numpy as np
import csv
from os.path import expanduser
import os
import matplotlib.pyplot as plt

# MODEL VARIABLES  -----------------------------------------------------------
iterations = 80  # number of iteration, originally set to 80 in the paper
# To get a better-looking figure i=10,000 is recommended (Christensen uses 80,000).
# However, this requires some computational power and/or time.

m = 30  # number of dimensions  (1991 paper has m=30)
n = 50  # number of people      (1991 paper has n=50)

p2 = 0.1  # Org code learning rate, I suggest changing p2 to 0.1, 0.5, and 0.9
p3 = 0  # employee turnover
p4 = 0  # environmental turbulence

time_limit = 100  # is only binding when either p3 or p4 are positive

# P1 PARAMETER -------------------------------------------------------
# The list forming the parmeter space for the socializatoin rate
# I suggest iterating over P1_list from 0.1 to 0.9.

P1_list = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

# PREPARING THE OUTPUT MATRIX ------------------------------------------------
Output = np.zeros(len(P1_list))
scenario = 0  # counter to report progress

# SIMULATION  ----------------------------------------------------------------
c_p1 = 0  # counter for recording rows in the OUTPUT file
for p1 in P1_list:  # over the speed of learning of individuals
    scenario += 1
    Equilibrium_knowledge = np.zeros(iterations).astype(int)
    for i in np.arange(iterations):
        print("Scenario: ", str(scenario), " out of ", str(len(P1_list)), ", iteration: ", str(i))
        '''
        "Thus, the process begins with an organizational code
        characterized by neutral beliefs on all dimensions and a set of
        individuals with varying beliefs that exhibit, on average, no
        knowledge. Over time, the organizational code affects the beliefs
        of individuals, even while it is being affected by those
        beliefs. The beliefs of individuals do not affect the beliefs of
        other individuals directly but only through affecting the
        code. The effects of reality are also indirect.  Neither the
        individuals nor the organizations experience reality.
        Improvement in knowledge comes by the code mimicking the
        beliefs (including the false beliefs) of superior individuals and
        by individuals mimicking the code (including its false beliefs)."
        March 1991: 75
        '''
        # Generating initial objects
        external_reality = (2*np.random.binomial(1, 0.5, m))-1
        '''
        "(1) There is an external reality that is independent of beliefs
        about it. Reality is described as having m dimensions, each of
        which has a value of 1 or -1. The (independent) probability that
        any one dimension will have a value of 1 is 0.5." March 1991: 74
        '''
        beliefs = np.random.choice(3, (m, n))-1
        org_code = np.zeros(m).astype(int)  # neutral beliefs on all dimensions
        '''
        "(2) At each time period, beliefs about reality are held by each of
        n individuals in an organization and by an organizational code of
        received truth. For each of the m dimensions of reality, each
        belief has a value of 1, 0, or -1. This value may change over time"
        March 1991: 74
        '''
        
        # we will need the following two variables for the termination condition
        equilibrium = 0  # are code and each individual belief the same
        period = 0  # to check when the termination condition is activated
        
        while equilibrium == 0:   
            '''
            # employee turnover =====
            "Suppose that each time period each individual has a probability,
            p3, of leaving the organization and being replaced by a new
            individual with a set of naive beliefs described by an m-tuple,
            having values equal to 1, 0, or -1, with equal probabilities."
            March 1991: 78
            '''
            for worker in np.arange(n):
                if np.random.rand() < p3:
                    beliefs[:, worker] = np.random.choice(3, m)-1
            
            '''
            # environmental turbulence =====
            "Suppose that the value of any given dimension of realitv shifts.
            (from 1 to - 1 or - 1 to 1) in agiven time-period with
            probability p4." March 1991: 79
            '''
            for dim in np.arange(m):
                if np.random.rand() < p4:
                    external_reality[dim] = external_reality[dim]*(-1)
            
            '''
            # learning =====
            From Michael Christensen's note, the model has three steps
            1. Individuals act
            2. Individuals socialize
            3. Organization learns
            '''
            
            # === PROCESS OF ORGANIZATIONAL LEARNING  =========
            # 1. INDIVIDUALS ACT
            Performance_n = np.zeros(n).astype(int)  # performance of individuals
            Actions_n = np.zeros((m, n)).astype(int)
            for n_ in np.arange(n):  # iterating over individuals
                Action = np.zeros(m).astype(int)
                for m_ in np.arange(m):  # iterating over dimensions
                    if beliefs[m_, n_] != 0:
                        Action[m_] = np.copy(beliefs[m_, n_])
                    else:  # if in doubt , flip a coin
                        Action[m_] = int(np.random.choice(2)*2-1)
                
                Performance_n[n_] = int(sum(Action*external_reality))  # MC note
                Actions_n[:, n_] = np.copy(Action)
                
              
            # 2. INDIVIDUALS SOCIALIZE
            '''
            "(3) Individuals modify their beliefs continuously as a
            consequence of socialization into the organization and education
            into its code of beliefs. Specifically, if the code is 0 on a
            particular dimension, individual belief is not affected. In each
            period in which the code differs on any particular dimension from
            the belief of an individual, individual belief changes to that of
            the code with probability, p1. Thus, p1 is a parameter reflecting
            the effectiveness of socialization, i.e., learning from the code.
            Changes on the several dimensions are assumed to be independent
            of each other." March 1991: 74
            
            "Socialization gives each component of each agent’s beliefs a
            chance (p1) to align with the organizational code if the code
            on the particular dimension is nonzero. If an agent changes a
            certain belief, the change is gradual or in steps, in the sense
            that a value of zero agent belief will align completely with
            the organizational code whereas an agent belief completely
            misaligned with the organizational code will turn zero. The
            gradual change allows agents to return to the experimenting
            phase." Christensen 2015: 4
            
            This last point that individuals socialize gradually is very
            important. Without it the results will look like Figure 1, but
            the values will not match exactly.
            '''

            for n_ in np.arange(n):  # iterating over individuals
                for m_ in np.arange(m):  # iterating over dimensions
                    if np.random.uniform() < p1:
                        if int(org_code[m_]) == 1:  # gradual socialization
                            if beliefs[m_, n_] == -1:
                                beliefs[m_, n_] = 0
                            elif beliefs[m_, n_] == 0:
                                beliefs[m_, n_] = 1
                        elif int(org_code[m_]) == -1:  # gradual socialization
                            if beliefs[m_, n_] == 1:
                                beliefs[m_, n_] = 0
                            elif beliefs[m_, n_] == 0:
                                beliefs[m_, n_] = -1
            
            # 3. ORGANIZATIONAL CODE LEARNS
            
            # First we look at how well the organization performs
            # This part is not described in the paper and Footnote 1 is
            # in fact incorrect in representing how the model works. The
            # model does not compare its knowledge to that of employees,
            # but rather employees and the code execute actions and observe
            # performance. The performance is compared, not the underlying
            # knowledge and thus a lucky (rather than smart) individual
            # may be considered to be superior.
            
            Org_knowledge_score = int(sum(org_code*external_reality))  # Christensen note

            '''
            "(4) At the same time, the organizational code adapts to the
            beliefs of those individuals whose beliefs correspond with
            reality on more dimensions than does the code. The probability
            that the beliefs of the code will be adjusted to conform to the
            dominant belief within the superior group on any particular
            dimension depends on the level of agreement among individuals in
            the superior group and on p2.1 Thus, P2 is a parameter reflecting
            the effectiveness of learning by the code. Changes on the several
            dimensions are assumed to be independent of each other."
            March 1991: 74
            
            Note: here is where the paper departs from the code used for the
            1991 paper. Instead of looking at the knowledge of individuals,
            the organization observes actual outcomes of actions and actions
            themselves individuals take and compares it to its own knowledge.
            Here the organization doesn't look at its own performance but simply
            at on how many dimensions it is correct about reality.
            '''
            # Now each individual acts and we record their performance
            # and note if their performance is superior to that of the code

            Superior = np.zeros(n).astype(int)  # set up a list of superior individuals
            for n_ in np.arange(n):
                if int(Performance_n[n_]) > Org_knowledge_score:
                    Superior[n_] = 1

            # Now the code is learning from the superior individuals
            '''
            "More precisely, if the code is the same as the majority
            view among those individuals whose overall knowledge score is
            superior to that of the code, the code remains unchanged. If
            the code differs from the majority view on a particular dimension
            at the start of a time period, the probability that it will be
            unchanged at the end of period is (1 - P2)^k, where k (k > 0) is
            the number of individuals (within the superior group) who differ
            from the code on this dimension minus the number who do not.
            This formulation makes the effective rate of code learning
            dependent on k, which probably depends on n. In the present
            simulations, n is not varied."
            footnote 1, March 1991: 74
            '''
            
            if int(sum(Superior))>0:  # if there are any superior individuals
                Actions_superior = np.zeros((m, int(sum(Superior)))).astype(int)
                counter = 0
                for n_ in np.arange(n):
                    if Superior[n_] == 1:
                        Actions_superior[:, counter] = np.copy(Actions_n[:, n_])
                        counter += 1
                for m_ in np.arange(m):  # over dimensions of reality
                    # first, if majority's view is different from the code
                    if sum(Actions_superior[m_, :]==-1)>sum(Actions_superior[m_, :]==1):
                        if org_code[m_] != -1:  # if org code different
                            k = sum(Actions_superior[m_, :]==-1)-sum(Actions_superior[m_, :]==1)
                            if np.random.uniform()>round((1-p2), 2)**int(k):  # to change
                                org_code[m_] = -1  # learns
                    elif sum(Actions_superior[m_, :]==1)>sum(Actions_superior[m_, :]==-1):
                        if org_code[m_] != 1:  # if org code different
                            k = sum(Actions_superior[m_, :]==1)-sum(Actions_superior[m_, :]==-1)
                            if np.random.uniform()>round((1-p2), 2)**int(k):  # to change
                                org_code[m_] = 1  # learns
                    elif sum(Actions_superior[m_, :]==-1)==sum(Actions_superior[m_, :]==1):
                        org_code[m_] = org_code[m_]

                    # The last condition is superfluous here, but kept for
                    # consistency.
                    # As per 1991 article, when the superior employees are equally
                    # split regarding the best course of action,
                    # then there is no change. Strict majority is required
            
            
            # TERMINATION  ===============================================
            '''
            "Within this closed system, the model yields time paths of organizational
            and individual beliefs, thus knowledge levels, that depend
            stochastically on the initial conditions and the parameters affecting
            learning.  The basic features of these histories can be summarized
            simply: Each of the adjustments in beliefs serves to eliminate
            differences between the individuals and the code.  Consequently, the
            beliefs of indi- viduals and the code converge over time. As individuals
            in the organization become more knowledgeable, they also become more
            homogeneous with respect to knowl- edge. An equilibrium is reached at
            which all individuals and the code share the same (not necessarily
            accurate) belief with respect to each dimension. The equilibrium is
            stable." march 1991: 75
            '''
            
            # Our goal here is to check if there is any possiblity for
            # the code to learn from the individuals or for the idividuals
            # to learn from the code. If neither is possible, the learning
            # dynamics stops. Most of the time the code takes about 80-130
            # periods for p1 = 0.1 before it reaches equilibrium, but it
            # can also reach 500+ rounds in some cases. 
            
            if p3 + p4 == 0:  # if no turnover or turbulence
                code_opportunity = 1  # let's assume code can still learn from individuals
                Potential_superiors = np.zeros(n).astype(int)  # set up a list of superior individuals
                for n_ in np.arange(n):
                    Individual_score = int(sum(beliefs[:, n_]*external_reality)) + int(sum(beliefs[:, n_]==0))
                    # Here we calculate maximum possible performance of each individual
                    # based on his personal beliefs.
                    if Individual_score > int(Org_knowledge_score):
                        Potential_superiors[n_] = 1
                if int(sum(Potential_superiors)) == 0:  # if there are no individuals
                    code_opportunity = 0
                
                # Opportunity for an individual to learn from the code ===========
                # Now let's see if it is still possible for individuals to learn from
                # the code. Here some may say that this condition alone should be
                # sufficient. However, there exists a very small probability that
                # all agents have the same beliefs as the code, but on one or more
                # dimensions the shared belief is equal to 0. That means that
                # individuals won't learn from the code, but the code can still
                # learn from the actions of individuals. Probability of this
                # happening decreases with n.
                # This means that we still need the previous condition.
                
                individual_opportunity = 0  # let's assume there is no opportunity for individuals
                for n_ in np.arange(n):
                    if int(sum(beliefs[:, n_]==org_code)) != m:
                        # we are checking if each individual differs from the code
                        individual_opportunity = 1
                
                if code_opportunity == 0 and individual_opportunity == 0:
                    Equilibrium_knowledge[i] = sum(org_code*external_reality)
                    equilibrium = 1  # we stop the simulation
                    print('Last period: ' + str(period))# from note Christensen
            else:
                # for situations where p3 or p4 are positive, the equilibrium
                # may never be reached. Hence, we add the time limit to be
                # activated when at least one of these parameters is positive
                if period == time_limit:
                    equilibrium = 1

            period += 1
        # Recording knowledge
        '''
        "Thus, the process begins with an organizational code characterized
        by neutral beliefs on all dimensions and a set of individuals with
        varying beliefs that exhibit, on average, no knowledge." March 1991: 75
        '''
        
    Output[c_p1] = np.mean(Equilibrium_knowledge)/float(m)
    c_p1 += 1
        
# Plot the average equilibrium knowledge (Figure 1)
plt.figure(1, facecolor='white', figsize=(8, 6))
plt.plot(P1_list, Output, color='blue', linewidth=1, label='p2='+str(p2))
plt.legend(loc=1,prop={'size':10})
plt.title('Recreation of March 1991 Figure 1', size=12)
plt.xlabel('p1', size=12)
plt.ylabel('average equilibrium knowledge', size=12)

# SAVING RESULTS TO CSV ----
# here we save only the average equilibrium knowledge required to Figure 1
_home = expanduser('~')
path_A = _home + '\\March_1991\\'
if not os.path.exists(path_A):
    os.makedirs(path_A)

filename = (path_A + 'March_1991_p1_' + str(p1) + '_p2_' + str(p2) +
            '_p3_' + str(p3) + '_p4_' + str(p4) + '_t_limit_' + str(time_limit) +
            '_i_' + str(iterations) + '.csv')
results = open(filename, 'w', newline='')
writer = csv.writer(results)
writer.writerow(['p2='+str(p2)])
print(Output)

for row in Output:
    writer.writerow([row])
results.close()

# END OF LINE
