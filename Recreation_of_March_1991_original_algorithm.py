"""
March 1991

@author: Maciej Workiewicz (ESSEC)
date: September 25, 2019 (This version: August 14, 2023)

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

The code has been tested on Anaconda Python 3.10
"""

import os
from os.path import expanduser
import numpy as np
from random import choices
from random import choice
from tabulate import tabulate
import csv
import matplotlib.pyplot as plt
from cycler import cycler

# MODEL VARIABLES  -----------------------------------------------------------
iterations = 10_000  # number of iteration, originally set to 80 in the paper
# To get a better-looking figure i=10,000 is recommended (Christensen uses 80,000).
# However, this requires some computational power and/or time.

m = 30  # number of dimensions  (1991 paper has m=30)
n = 50  # number of people      (1991 paper has n=50)

p3 = 0  # employee turnover
p4 = 0  # environmental turbulence

time_limit = 100  # is only binding when either p3 or p4 are positive

# P1 & P2 PARAMETER -------------------------------------------------------
# The parameter spaces for the socialization rate (p1) and code learning rate (p2)
# I suggest iterating over P1_list from 0.1 to 0.9.
# I suggest changing p2 to 0.1, 0.5, and 0.9

P1_list = np.array([.1, .2, .3, .4, .5, .6, .7, .8, .9])
P2_list = np.array([.1, .5, .9])

# PREPARING THE OUTPUT MATRIX ------------------------------------------------
Output = np.zeros([len(P1_list), len(P2_list)])
scenario = 0  # counter to report progress

# SIMULATION  ----------------------------------------------------------------
P1P2_list = np.array(np.meshgrid(P1_list, P2_list)).T.reshape(-1, 2)
P1P2_id_list = np.array(np.meshgrid(range(len(P1_list)), range(len(P2_list)))).T.reshape(-1, 2)
for P1P2_id in P1P2_id_list:  # over the speed of learning of individuals
    p1 = P1_list[P1P2_id[0]]
    p2 = P2_list[P1P2_id[1]]
    scenario += 1
    Equilibrium_knowledge = np.zeros(iterations, dtype=int)
    for i in np.arange(iterations):
        print("Scenario", str(scenario), "out of", str(len(P1P2_id_list)), ": (p1, p2) = (", str(p1), ",", str(p2), ")", "\t( iteration:", str(i), "/", iterations, ")")

        """
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
        """
        # Generating initial objects
        external_reality = np.array(choices([-1, 1], k=m))
        """
        "(1) There is an external reality that is independent of beliefs
        about it. Reality is described as having m dimensions, each of
        which has a value of 1 or -1. The (independent) probability that
        any one dimension will have a value of 1 is 0.5." March 1991: 74
        """
        beliefs = np.reshape(choices([-1, 0, 1], k=m * n), (m, n))
        org_code = np.zeros(m, dtype=int)  # neutral beliefs on all dimensions
        """
        "(2) At each time period, beliefs about reality are held by each of
        n individuals in an organization and by an organizational code of
        received truth. For each of the m dimensions of reality, each
        belief has a value of 1, 0, or -1. This value may change over time"
        March 1991: 74
        """

        # we will need the following two variables for the termination condition
        is_equilibrium = False  # to check if code and individuals have no learning opportunity
        period = 0  # to check when the termination condition is activated

        while not is_equilibrium:
            """
            # employee turnover =====
            "Suppose that each time period each individual has a probability,
            p3, of leaving the organization and being replaced by a new
            individual with a set of naive beliefs described by an m-tuple,
            having values equal to 1, 0, or -1, with equal probabilities."
            March 1991: 78
            """
            for n_ in np.arange(n):
                if np.random.rand() < p3:
                    beliefs[:, n_] = np.array(choices([-1, 0, 1], m))

            """
            # environmental turbulence =====
            "Suppose that the value of any given dimension of reality shifts.
            (from 1 to - 1 or - 1 to 1) in a given time-period with
            probability p4." March 1991: 79
            """
            for dim in np.arange(m):
                if np.random.rand() < p4:
                    external_reality[dim] = -external_reality[dim]

            """
            # learning =====
            From Michael Christensen's note, the model has three steps
            1. Individuals act
            2. Individuals socialize
            3. Organization learns
            """

            # === PROCESS OF ORGANIZATIONAL LEARNING  =========
            # 1. INDIVIDUALS ACT

            Knowledge_n = np.zeros(n, dtype=int)  # performance of individuals
            Actions_n = np.zeros((m, n), dtype=int)
            for n_ in np.arange(n):  # iterating over individuals
                Action = np.copy(beliefs[:, n_])
                for m_ in np.arange(m):  # iterating over dimensions
                    if beliefs[m_, n_] == 0:  # if in doubt , flip a coin
                        Action[m_] = choice([-1, 1])
                Knowledge_n[n_] = sum(Action * external_reality)  # Christensen note
                # Note that wrong beliefs (except ignorance) reduce knowledge.
                Actions_n[:, n_] = Action

            # 2. INDIVIDUALS SOCIALIZE
            """
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
            
            "Socialization gives each component of each agent's beliefs a
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
            """

            pregenerated_random = np.random.uniform(low=0, high=1, size=n * m)
            pregenerated_random_index = 0
            # Pregenerating n*m random numbers is usually faster than generating one by one.
            for n_ in np.arange(n):  # iterating over individuals
                for m_ in np.arange(m):  # iterating over dimensions
                    if pregenerated_random[pregenerated_random_index] < p1:
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
                    pregenerated_random_index += 1

            # 3. ORGANIZATIONAL CODE LEARNS
            """
            First we look at how well the organization performs
            This part is not described in the paper and Footnote 1 is
            in fact incorrect in representing how the model works. The
            model does not compare its knowledge to that of employees,
            but rather employees and the code execute actions and observe
            performance. The performance is compared, not the underlying
            knowledge and thus a lucky (rather than smart) individual
            may be considered to be superior.
            """

            Org_knowledge_score = sum(org_code * external_reality)  # Christensen note

            """
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
            """
            # Now each individual acts and we record their performance
            # and note if their performance is superior to that of the code

            isSuperior = Knowledge_n > Org_knowledge_score  # set up a list of superior individuals

            # Now the code is learning from the superior individuals
            """
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
            """

            if any(isSuperior):  # if there are any superior individuals
                Actions_superior = Actions_n[:, isSuperior]
                Sum_Actions_superior = np.sum(Actions_superior, axis=1)  # sum of actions across superior individuals
                Majority_view_superior = np.clip(Sum_Actions_superior, -1, 1)
                pregenerated_random = np.random.uniform(low=0, high=1, size=m)
                for m_ in np.arange(m):  # over dimensions of reality
                    if Majority_view_superior[m_] == 0:
                        org_code[m_] = org_code[m_]
                        # This condition is superfluous here, but kept for consistency.
                        # As per 1991 article, when the superior employees are equally
                        # split regarding the best course of action, there is no change.
                        # Strict majority is required as follows:
                    elif Majority_view_superior[m_] != org_code[m_]:
                        # If the strict majority view is different from the code
                        k = abs(Sum_Actions_superior[m_])  # How much one opinion is stronger than another
                        if pregenerated_random[m_] > round((1 - p2), 2) ** k:  # probability to change
                            org_code[m_] = Majority_view_superior[m_]  # learns

            # TERMINATION  ===============================================
            """
            "Within this closed system, the model yields time paths of organizational
            and individual beliefs, thus knowledge levels, that depend
            stochastically on the initial conditions and the parameters affecting
            learning.  The basic features of these histories can be summarized
            simply: Each of the adjustments in beliefs serves to eliminate
            differences between the individuals and the code.  Consequently, the
            beliefs of individuals and the code converge over time. As individuals
            in the organization become more knowledgeable, they also become more
            homogeneous with respect to knowl- edge. An equilibrium is reached at
            which all individuals and the code share the same (not necessarily
            accurate) belief with respect to each dimension. The equilibrium is
            stable." March 1991: 75
            """

            # Our goal here is to check if there is any possibility for
            # the code to learn from the individuals or for the individuals
            # to learn from the code. If neither is possible, the learning
            # dynamics stops. Most of the time the code takes about 80-130
            # periods for p1 = 0.1 before it reaches equilibrium, but it
            # can also reach 500+ rounds in some cases.

            if p3 + p4 == 0:  # if no turnover or turbulence
                code_opportunity = True  # let's assume code can still learn from individuals
                Potential_superiors = np.zeros(n, dtype=bool)  # set up a list of superior individuals
                for n_ in np.arange(n):
                    Individual_score = sum(beliefs[:, n_] * external_reality) + sum(beliefs[:, n_] == 0)
                    # `Individual_score` is the maximum possible performance of the `n_`th individual,
                    #  determined by their personal beliefs. It comprises two components:
                    #  * Fixed Component: Knowledge level of non-zero beliefs
                    #  * Random Component: Count of zero beliefs, which result in a random action (See `INDIVIDUALS ACT`)
                    if Individual_score > Org_knowledge_score:
                        Potential_superiors[n_] = True
                if not any(Potential_superiors):  # if there are no superior individual
                    code_opportunity = False

                # Opportunity for an individual to learn from the code ===========
                # Now let's see if it is still possible for individuals to learn from
                # the code, given that individuals only learn on non-zero code dimensions.
                # When there is any 0 left in the code, the "potential-superior"
                # condition is necessary. In this case, individuals won't learn from
                # the code, but the code can still learn from the actions of individuals.
                # Probability of this happening decreases with n.
                # This means that we still need the previous condition.

                individual_opportunity = False  # let's assume there is no opportunity for individuals
                for n_ in np.arange(n):
                    if any((beliefs[:, n_] != org_code) & (org_code != 0)):
                        # we are checking if any individual differs from the code on non-zero dimensions
                        individual_opportunity = True
                if not (code_opportunity or individual_opportunity):
                    Equilibrium_knowledge[i] = sum(org_code * external_reality)  # Christensen note
                    is_equilibrium = True  # we stop the simulation
                    print('Last period: ' + str(period))  # from note Christensen
            else:
                # for situations where p3 or p4 are positive, the equilibrium
                # may never be reached. Hence, we add the time limit to be
                # activated when at least one of these parameters is positive
                if period == time_limit:
                    is_equilibrium = True

            period += 1
        # Recording knowledge
        """
        "Thus, the process begins with an organizational code characterized
        by neutral beliefs on all dimensions and a set of individuals with
        varying beliefs that exhibit, on average, no knowledge." March 1991: 75
        """
        Output[P1P2_id[0], P1P2_id[1]] = np.mean(Equilibrium_knowledge) / float(m)

# PRINTING RESULTS
p2_label = ['p2=' + str(p2) for p2 in P2_list]
print(tabulate(Output, headers=p2_label))

# SAVING RESULTS
_home = expanduser('~')
path_A = _home + '\\March_1991\\'
if not os.path.exists(path_A):
    os.makedirs(path_A)
filename = (path_A + 'March_1991' +
            '_p1_' + str(min(P1_list)) + '-' + str(max(P1_list)) +
            '_p2_' + str(min(P2_list)) + '-' + str(max(P2_list)) +
            '_p3_' + str(p3) + '_p4_' + str(p4) + '_t_limit_' + str(time_limit) +
            '_i_' + str(iterations))
# *.csv: we save only the average equilibrium knowledge required to Figure 1
with open(filename + '.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    # writing the header (p2 values)
    writer.writerow(p2_label)
    # writing the output values
    writer.writerows(Output)
    file.close()
# *.png: we plot the average equilibrium knowledge (Figure 1)
p2_label = [r'$p_2=$' + str(p2) for p2 in P2_list]
plt.figure(1, figsize=(6, 8))
if len(P2_list) > 4:
    # Adopted from H. Ranocha's work: https://ranocha.de/blog/colors/
    line_cycler = (cycler(color=['#E69F00', '#56B4E9', '#009E73', '#0072B2', '#D55E00', '#CC79A7', '#F0E442']) +
                   cycler(linestyle=['-', '--', '-.', ':', '-', '--', '-.']))
else:
    # Adopted from March (1991)
    line_cycler = (cycler(color=['black', 'black', 'black', 'black']) +
                   cycler(linestyle=['-', '--', ':', '-.']))
plt.rc('axes', prop_cycle=line_cycler)
plt.rc('font', family='sans-serif')
plt.plot(P1_list, Output, linewidth=1, label=p2_label)
plt.ylim(.70, 1.00)
plt.title('(Recreated from March 1991)\n' +
          r'Figure 1. Effect of Learning Rate ($p_1$, $p_2$) on Equilibrium Knowledge' + '\n' +
          r'$M=$' + str(m) + r'; $N=$' + str(n) + '; ' + str(iterations) + ' iterations', size=10)
plt.xlabel(r'SOCIALIZATION RATE ($p_1$)', size=10)
plt.ylabel('AVERAGE EQUILIBRIUM KNOWLEDGE', size=10)
plt.legend(loc=1, prop={'size': 10}, frameon=False)
plt.savefig(fname=filename + '.png', transparent=True)

# END OF LINE
