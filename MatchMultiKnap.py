from docplex.mp.model import Model


class MatchSolver:
    def __init__(self, slab_data_, order_data_):
        self.slab_data = [com.tolist() for com in slab_data_]
        self.order_data = [com.tolist() for com in order_data_]
        # print("slab data in solver: ", self.slab_data)
        # print("order data in solver: ", self.order_data)
        self.M = len(self.slab_data)    # 板坯数量
        self.N = len(self.order_data)   # 合同数量
        # 模型参数
        self.match_pro1 = 30.0  # 钢级相等
        self.match_pro2 = 1.5  # 板坯钢级大于合同钢级
        self.match_pro3 = 0.2  # 板坯钢级小于合同钢级
        self.self_design = 0.1
        self.match_mat = self.get_match_cost()
        # print(self.match_mat)

    def get_match_cost(self):
        # 计算模型目标相关参数
        # 用户合同匹配费用： shape:[M, N]
        c1 = list()
        for i in range(self.M):
            c1_order_i = list()
            for j in range(self.N):
                if self.order_data[j][0] < self.slab_data[i][0]:
                    c1_order_i.append(-99999.9)
                elif self.order_data[j][1] == self.slab_data[i][1]:
                    #c1_order_i.append(self.slab_data[i][0] * (self.match_pro1 - self.self_design))
                    c1_order_i.append(self.slab_data[i][0] * (self.match_pro1 ))
                elif self.order_data[j][1] < self.slab_data[i][1]:
                    #c1_order_i.append(self.slab_data[i][0] * (self.match_pro2 - self.self_design))
                    c1_order_i.append(self.slab_data[i][0] * (self.match_pro2 ))
                else:
                    #c1_order_i.append(self.slab_data[i][0] * (self.match_pro3 - self.self_design))
                    c1_order_i.append(self.slab_data[i][0] * (self.match_pro3))
            c1.append(c1_order_i)
        return c1

    def solve_model(self):
        my_model = Model("multi-knapsack problem")
        X = my_model.binary_var_matrix(keys1=list(range(self.M)), keys2=list(range(self.N)), name='X')
        # 不确定M和N谁大
        for i in range(self.M):
            my_model.add_constraint(my_model.sum([X[i, j] for j in range(self.N)]) <= 1)
        '''for j in range(self.N):
            my_model.add_constraint(my_model.sum([X[i, j] for i in range(self.M)]) <= 1)'''
        for j in range(self.N):
            slab_order_weight_list = list()
            for i in range(self.M):
                slab_order_weight_list.append(self.slab_data[i][0] * X[i, j])
            my_model.add_constraint(abs(self.order_data[j][0]) - my_model.sum(slab_order_weight_list) >= 0)
        my_model.maximize(my_model.sum([self.match_mat[i][j] * X[i, j] for i in range(self.M) for j in range(self.N)]))
        sln = my_model.solve()
        if sln is None:
            print("合同欠量：")
            for i in range(self.N):
                print(self.order_data[i][0])

        result1 = [[int(sln[X[i, j]]) for i in range(self.M)] for j in range(self.N)]    # [N, M]
        obj = sln.get_objective_value() + sum([self.slab_data[i][0] * self.slab_data[i][1] * self.self_design for i in range(self.M)])
        # print(result1)
        return [list(map(round, l)) for l in result1], obj





