from mimetypes import init
import Toolkit as tk
import ToyModel as tm




agent1=tk.Agent("agent1",
                model=tm.Toy_Model_NE_1,
                actor_network=tk.DDPGActor,
                critic_network=tk.DDPGCritic,
                buffer=tk.Memory(max_size=100000),
                observables=['agent1','agent2','S',"A","B"],
                actions=['EX_A_sp1','EX_B_sp1'],
                gamma=0.999,
                update_batch_size=32,
                )

agent2=tk.Agent("agent2",
                model=tm.Toy_Model_NE_2,
                actor_network=tk.DDPGActor,
                critic_network=tk.DDPGCritic,
                observables=['agent1','agent2','S',"A","B"],
                actions=['EX_A_sp2','EX_B_sp2'],
                buffer=tk.Memory(max_size=100000),
                gamma=0.999,
                update_batch_size=32,
)

agents=[agent1,agent2]

env=tk.Environment(name="Toy-NECOM",
                    agents=agents,
                    extracellular_reactions=[],
                    initial_condition={"S":100,"agent1":0.1,"agent2":0.1},
                    inlet_conditions={"S":10},
                    max_c={'S':100,
                           'agent1':10,  
                           'agent2':10,
                           'A':10,
                           'B':10,})
env.reset()
Batch=env.batch_step(env.generate_random_c(40))
for i in Batch:
    for ind,ag in enumerate(env.agents):
        ag.buffer.push((i[0][ag.observables],i[1][ind],i[2][ind],i[3][ag.observables]))
env.step()
print(Batch)
