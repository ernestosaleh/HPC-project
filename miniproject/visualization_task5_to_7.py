#Naive way to do the visualizations
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import seaborn as sns
import scipy


# Task 5 -----------------------------------------------------------------
#retrieve data
with open('results/task5_result.txt', 'r') as file:
    task5_content = file.read()

#process data
task5_content =dict(eval(task5_content))
result = np.array([np.array([key,val]) for key,val in task5_content.items()])
n_processors, exec_times = result.T
n_processors = n_processors.astype(np.int8)

# #compute speedup
speedup = exec_times[0]/exec_times

#visualize data
#Execution time plot
fig, ax = plt.subplots(1,1, figsize = (10,4))
ax.plot(n_processors, exec_times)
ax.set(title=f'Execution time vs. number of processors', xlabel='number of processors', ylabel='Time (s)')
ax.grid(True)
plt.savefig(f'images/task5/task5_execution_time.png', bbox_inches='tight')
plt.close()
def amdahl(P, F):
    return 1 / (F/P + (1 - F))

params, _ = scipy.optimize.curve_fit(amdahl, n_processors, speedup, bounds=(0, 1))
F_opt = params[0]

fig, ax = plt.subplots(1,1, figsize = (10,4))
ax.plot(n_processors, speedup, linestyle='-', marker='o', label='')
ax.plot(n_processors, [amdahl(p, F_opt) for p in n_processors], '--', label=f'p={F_opt:.2f}')
ax.set(title=f'Speedup vs. number of processors', xlabel='number of processors', ylabel='Speedup')
ax.legend()
ax.grid(True)
plt.savefig(f'images/task5/task5_amdahl.png', bbox_inches='tight')
plt.close()


# Task 6 -----------------------------------------------------------------
#retrieve data
with open('results/task6_result.txt', 'r') as file:
    task6_content = file.read()
    
#process data
task6_content =dict(eval(task6_content))
result = np.array([np.array([key,val]) for key,val in task6_content.items()])
n_processors, exec_times = result.T
n_processors = n_processors.astype(np.int8)

# #compute speedup
speedup = exec_times[0]/exec_times

#visualize data
#Execution time plot
fig, ax = plt.subplots(1,1, figsize = (10,4))
ax.plot(n_processors, exec_times)
ax.set(title=f'Execution time vs. number of processors', xlabel='number of processors', ylabel='Time (s)')
ax.grid(True)
plt.savefig(f'images/task6/task6_execution_time.png', bbox_inches='tight')
plt.close()
def amdahl(P, F):
    return 1 / (F/P + (1 - F))

params, _ = scipy.optimize.curve_fit(amdahl, n_processors, speedup, bounds=(0, 1))
F_opt = params[0]

fig, ax = plt.subplots(1,1, figsize = (10,4))
ax.plot(n_processors, speedup, linestyle='-', marker='o', label='')
ax.plot(n_processors, [amdahl(p, F_opt) for p in n_processors], '--', label=f'p={F_opt:.2f}')
ax.set(title=f'Speedup vs. number of processors', xlabel='number of processors', ylabel='Speedup')
ax.legend()
ax.grid(True)
plt.savefig(f'images/task6/task6_amdahl.png', bbox_inches='tight')
plt.close()
plt.show()


# Task 5vs6 -----------------------------------------------------------------
with open('results/task5_result.txt', 'r') as file:
    task5_content = file.read()
    task5_content =dict(eval(task5_content))

with open('results/task6_result.txt', 'r') as file:
    task6_content = file.read()
    task6_content =dict(eval(task6_content))
    
fig, ax = plt.subplots(1,1, figsize = (10,4)) 
#process data

tasks = dict(static = task5_content,
             dynamic = task6_content)

speedups = {}

for task in tasks.keys():
    task_content=tasks[task]
    result = np.array([np.array([key,val]) for key,val in task_content.items()])
    n_processors, exec_times = result.T
    n_processors = n_processors.astype(np.int8)
    result = np.array([np.array([key,val]) for key,val in task_content.items()])
    n_processors, exec_times = result.T
    n_processors = n_processors.astype(np.int8)
    speedup = exec_times[0]/exec_times
    speedups[task] = (n_processors, speedup)
    #visualize data
    #Execution time plot
    ax.plot(n_processors, np.log10(exec_times), linestyle='-', label=task)
    
ax.set(title=f'Execution time vs. number of processors', xlabel='number of processors', ylabel='log10(Execution time) (s)')
ax.legend()
ax.grid(True)
plt.savefig(f'images/task6/task5vs6_exec.png', bbox_inches='tight')
plt.close()

    
###Plot speedup
# #compute experimental speedup
###Compute experimental parallelization
speedup = speedups['dynamic'][1]
def amdahl(P, F):
    return 1 / (F/P + (1 - F))
params, _ = scipy.optimize.curve_fit(amdahl, n_processors, speedup, bounds=(0, 1))
F_opt = params[0]
fig, ax = plt.subplots(1,1, figsize = (10,4))

for task in tasks.keys():
    ax.plot(*speedups[task], linestyle='-', marker='o', label=task)
ax.plot(n_processors, [amdahl(p, F_opt) for p in n_processors], '--', label=f'p={F_opt:.2f}')


ax.set(title=f'Speedup vs. number of processors', xlabel='number of processors', ylabel='Speedup')
ax.legend()
ax.grid(True)
plt.savefig(f'images/task6/task5vs6_speedup.png', bbox_inches='tight')
plt.close()


# Task 7 -----------------------------------------------------------------

#retrieve data
with open('results/task7_result.txt', 'r') as file:
    building_ids, exec_times = file.readlines()

building_ids, exec_times= eval(building_ids), np.array(eval(exec_times))

#visualize
#Execution time plot
fig, ax = plt.subplots(1,1, figsize = (10,4))
ax.bar(building_ids, exec_times)
ax.set(title=f'Execution Times, total: {exec_times.sum():.4f} s', xlabel='Building ID', ylabel='Time (s)')
ax.axhline(y=np.mean(exec_times), color='red', linestyle='--', linewidth=1)  # Add horizontal line
ax.text(18, np.mean(exec_times) + 0.1, f"Average: {np.mean(exec_times):.4f}", color='red', ha='center')
ax.set_xticks(list(range(len(building_ids))))
ax.set_xticklabels(building_ids, rotation=90)
plt.savefig(f'images/task7/task7_execution_time.png', bbox_inches='tight')
plt.close()