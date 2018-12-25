import numpy as np

from entity.workflow import Workflow

# time_reward_matrix = np.random.rand(7, 5)
time_reward_matrix = [[0.1270, 0.1746, 0.8191, 1.6822, 3.2659],
                      [0.1643, 0.2236, 1.0270, 2.0975, 4.0479],
                      [0.1640, 0.2235, 1.0255, 2.0948, 4.0519],
                      [0.1656, 0.2251, 1.0312, 2.1049, 4.0768],
                      [0.2008, 0.2737, 1.2588, 2.5725, 4.9698],
                      [0.2005, 0.2729, 1.2553, 2.5651, 4.9564],
                      [0.2011, 0.2743, 1.2619, 2.5784, 4.9790]]

task_size = [
    [14, 6, 22, 61, 26, 69, 21, 49, 72, 53, 85, 13, 61, 7, 64, 76, 47, 52, 90, 45, 96, 32, 63, 61, 58, 10, 74, 80, 39,
     0],
    [91, 85, 39, 74, 90, 10, 12, 89, 45, 33, 31, 86, 46, 74, 32, 88, 19, 48, 36, 79, 30, 50, 65, 63, 99, 0, 0,
     0, 0, 0],
    [81, 95, 71, 99, 9, 52, 85, 98, 22, 43, 76, 69, 76, 51, 85, 11, 40, 89, 26, 74, 89, 23, 86, 21, 58, 11, 49,
     56, 97, 26],
    [43, 90, 75, 11, 69, 28, 46, 46, 72, 30, 46, 37, 61, 13, 32, 21, 32, 89, 30, 55, 66, 81, 49, 24, 0, 0, 0,
     0, 0, 0],
    [29, 78, 9, 36, 49, 11, 62, 56, 44, 21, 84, 2, 52, 95, 48, 72, 47, 65, 6, 25, 90, 94, 47, 16, 41, 66, 58, 41, 46,
     52]]

cost_reward_matrix = [[0.021844, 0.0300312, 0.1408852, 0.2893384, 0.5617348],
                      [0.0157728, 0.0214656, 0.098592, 0.20136, 0.3885984],
                      [0.062976, 0.085824, 0.393792, 0.8044032, 1.5559296],
                      [0.0553104, 0.0751834, 0.3444208, 0.7030366, 1.3616512],
                      [0.03726848, 0.05079872, 0.23363328, 0.477456, 0.92239488],
                      [0.02005, 0.02729, 0.12553, 0.25651, 0.49564],
                      [0.07464832, 0.10182016, 0.46841728, 0.95710208, 1.8482048]]


class Env:
    def __init__(self, n_vm):
        self.n_vm = n_vm
        self.n_actions = self.n_vm
        self.n_features = 1 + 7  # task_type and vm_time
        self.n_task = 138
        self.dim_state = self.n_task
        # self.time_reward_matrix = np.random.rand(self.n_vm, 4)

        self.workflow = None
        self.task = None
        self.vm_time = None
        self.vm_cost = None
        self.released = None
        self.start_time = None
        self.task_exec = None
        self.state = None
        self.done = None
        # self.reward = None
        self.reset()

    def reset(self):
        self.workflow = [Workflow(i) for i in range(5)]
        self.vm_time = np.zeros(self.n_vm)
        self.vm_cost = np.zeros(self.n_vm)
        self.released = [[], [], [], [], []]
        self.start_time = np.zeros(self.n_task)
        self.task_exec = []
        self.state = np.ones(self.n_task)
        base = 0
        for i in range(len(self.workflow)):
            if i != 0:
                base += self.workflow[i - 1].size
            for j in range(len(self.workflow[i].precursor)):
                # print(self.workflow[i].precursor[j])
                idle = base + self.workflow[i].precursor[j]
                self.state[idle] = 0
        # print(self.state)

        cnt = 0
        for i in range(self.dim_state):
            if self.state[i] == 0:
                cnt += 1
        if cnt == 1 or cnt == 0:
            index = 0
        else:
            index = np.random.randint(cnt - 1)
        for i in range(self.dim_state):
            if self.state[i] == 0 and index != 0:
                index -= 1
            elif self.state[i] == 0 and index == 0:
                self.task = i
                break

        self.done = False
        # self.reward = 0

        return self.observation()

    def step(self, action):
        obs = []
        reward = []
        done = []

        self.set_action()
        # print('step')
        reward.append(self.rewards(action))
        obs.append(self.observation())
        done.append(self.is_done())

        return obs[0], reward[0], done[0]

    def has_value(self, arry, value):
        for i in range(len(arry)):
            if arry[i] == value:
                return True
        return False

    def release_node(self, task):
        # print(col)

        release = []
        count = 0
        belong = []

        for i in range(len(self.workflow)):
            if task < count + self.workflow[i].size:
                belong.append(i)
                belong.append(task - count)
                break
            count += self.workflow[i].size

        # print(belong)
        # print(self.scenario.workflows[belong[0]].structure)
        # print(self.scenario.node)
        self.released[belong[0]].append(belong[1])
        # print(self.scenario.node)

        back_node = []
        for i in range(self.workflow[belong[0]].size):
            if self.workflow[belong[0]].structure[belong[1]][i] == 1:
                back_node.append(i)

        # print(back_node)

        for i in range(len(back_node)):
            for j in range(self.workflow[belong[0]].size):
                if self.workflow[belong[0]].structure[j][back_node[i]] == 1 and not self.has_value(
                        self.released[belong[0]], j):
                    break
                elif j == self.workflow[belong[0]].size - 1:
                    release.append([belong[0], back_node[i]])
        # print(release)
        return release

    def set_action(self):
        self.state[self.task] = 1

        release = self.release_node(self.task)

        # print(release)
        if len(release) != 0:
            # cnt = 0
            for i in range(len(release)):
                cnt = 0
                if release[i][0] != 0:
                    for j in range(release[i][0]):
                        cnt += self.workflow[j].size
                cnt += release[i][1]
                # for i in range(7):
                self.state[cnt] = 0

        # for i in range(len(self.dim_state)):
        #     if self.state[i] == 0:
        #         self.task = i
        #         break

    def observation(self):
        # get task_type
        count = 0
        belong = []
        for i in range(len(self.workflow)):
            if self.task < count + self.workflow[i].size:
                belong.append(i)
                belong.append(self.task - count)
                break
            count += self.workflow[i].size
        # print(belong)
        task_type = self.workflow[belong[0]].subTask[belong[1]].task_type

        # TODO(hang): env.vm_cost
        return np.concatenate(([task_type], self.vm_time), 0)

    def time_reward(self, action):
        strategy = []
        last_makespan = max(self.vm_time)

        count = 0
        belong = []
        for i in range(len(self.workflow)):
            if self.task < count + self.workflow[i].size:
                belong.append(i)
                belong.append(self.task - count)
                break
            count += self.workflow[i].size

        strategy.append(belong[0])
        strategy.append(action + 1)

        task_type = self.workflow[belong[0]].subTask[belong[1]].task_type
        # print(task_type)

        # print(agent.state.pos, type)
        exec_time = time_reward_matrix[action][task_type]
        cost = cost_reward_matrix[action][task_type]
        self.vm_cost[action] += cost
        # size = task_size[belong[0]][belong[1]]
        # exec_time = exec_time * 2
        # print(exec_time)

        if self.vm_time[action] >= self.start_time[self.task]:
            strategy.append(self.vm_time[action])
            self.vm_time[action] += exec_time
            strategy.append(self.vm_time[action])
        else:
            strategy.append(self.start_time[self.task])
            self.vm_time[action] = self.start_time[self.task] + exec_time
            strategy.append(self.vm_time[action])

        self.task_exec.append(strategy)

        finish_time = self.vm_time[action]

        back_node = []
        for i in range(self.workflow[belong[0]].size):
            if self.workflow[belong[0]].structure[belong[1]][i] == 1:
                back_node.append(i)

        for i in range(len(back_node)):
            if finish_time > self.start_time[back_node[i]]:
                self.start_time[back_node[i]] = finish_time

        # print(vm_finish_time)
        # time = max(self.vm_time) - last_makespan

        cnt = 0
        for i in range(self.dim_state):
            if self.state[i] == 0:
                cnt += 1
        if cnt == 1 or cnt == 0:
            index = 0
        else:
            index = np.random.randint(cnt - 1)
        for i in range(self.dim_state):
            if self.state[i] == 0 and index != 0:
                index -= 1
            elif self.state[i] == 0 and index == 0:
                self.task = i
                break

        # return max(vm_finish_time)
        return last_makespan, exec_time
        # return pow((4.979 - reward) / 4.079, 2)

    def cost_reward(self, action):
        count = 0
        belong = []
        for i in range(len(self.workflow)):
            if self.task < count + self.workflow[i].size:
                belong.append(i)
                belong.append(self.task - count)
                break
            count += self.workflow[i].size
        task_type = self.workflow[belong[0]].subTask[belong[1]].task_type

        col = np.array(cost_reward_matrix)[:, task_type]
        worst = col[np.argmax(col)]
        best = col[np.argmin(col)]
        cost = cost_reward_matrix[action][task_type]
        self.vm_cost[action] += cost

        cnt = 0
        for i in range(self.dim_state):
            if self.state[i] == 0:
                cnt += 1
        if cnt == 1 or cnt == 0:
            index = 0
        else:
            index = np.random.randint(cnt - 1)
        for i in range(self.dim_state):
            if self.state[i] == 0 and index != 0:
                index -= 1
            elif self.state[i] == 0 and index == 0:
                self.task = i
                break

        return best, worst, cost

    def rewards(self, action):
        last_makespan, exec_time = self.time_reward(action)
        inc_makespan = max(self.vm_time) - last_makespan
        inc_makespan = round(inc_makespan, 4)
        # if inc_makespan == exec_time:
        #     return -0.5
        # else:
        # if pow((exec_time - inc_makespan) / exec_time, 3) < 0:
        #     print(exec_time,inc_makespan,max(self.vm_time),last_makespan)
        # print(pow((exec_time - inc_makespan) / exec_time, 3))

        # if np.sum(self.vm_cost) > 20:
        #     return -0.1

        return pow((exec_time - inc_makespan) / exec_time, 3)

        # TODO(hang): env.vm_cost
        # b_cost, w_cost, a_cost = self.cost_reward(action)
        #
        # # if w_cost == a_cost:
        # #     return -0.5
        # # else:
        #     # print(pow((w_cost - a_cost) / (w_cost - b_cost), 3))
        # return pow((w_cost - a_cost) / (w_cost - b_cost), 3)

    def is_done(self):
        # print(self.state)
        for i in self.state:
            if i != 1:
                return False
        return True
