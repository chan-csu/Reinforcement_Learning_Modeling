# Reinforcement_Learning_Modeling
----

Parsa Ghadermazi 1-31-2022
parsa96@colostate.edu

----

Reinforcement learning provides an ideal framework for finding optimal microbe-environment interactions. First project that uses this framework, as discussed, is going to be a simple case where there are two amylase producer e. coli in a chemostat with initial glucose and starch present in the reactor and the chhemostat is constantly fed with starch. The following shows the steps to see the effect of training agents with RL so that they come up with policies that leads to their long term survival:

## Step 1: Run a sterile chemostat
-----

this is only to show that what would be the behavior of the reactor in the absence of microbes

## Step 2: Run with a community of a cheater and a producer
----
As shown earlier, this community is not stable. I expect that the trained microbes should do better than this scenario

## Step 3: Train the microbes with RL Algorithm

Pay attention to the following diagram describing this step 

![Drag Racing](./overall.png)

Every state-action value will be evaluated for each policy using Monte Carlo Method. Return wou;d be the sum of a an agent concentration from that time until the end of that episode, maybe discounted or not. Policy improvement will be started after the returns are convegred to a good degree. The final optimal policy will be outputed

## Step 4: Data Analysis

Trained Agents will be tested in a community of two, and the profiles will be compared to see if RL has been successfull.