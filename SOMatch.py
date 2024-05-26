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
        self.match_pro1 = 10.0  # 钢级相等
        self.match_pro2 = 8.0  # 板坯钢级大于合同钢级
        self.match_pro3 = 6.0  # 板坯钢级小于合同钢级
        self.self_design = 1.0
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
                    c1_order_i.append(self.slab_data[i][0] * (self.match_pro1 - self.self_design))
                elif self.order_data[j][1] < self.slab_data[i][1]:
                    c1_order_i.append(self.slab_data[i][0] * (self.match_pro2 - self.self_design))
                else:
                    c1_order_i.append(self.slab_data[i][0] * (self.match_pro3 - self.self_design))
            c1.append(c1_order_i)
        return c1

    def solve_model(self):
        my_model = Model("assignment problem")
        X = my_model.continuous_var_matrix(keys1=list(range(self.M)), keys2=list(range(self.N)), name='X')
        # 不确定M和N谁大
        for i in range(self.M):
            my_model.add_constraint(my_model.sum([X[i, j] for j in range(self.N)]) <= 1)
        for j in range(self.N):
            my_model.add_constraint(my_model.sum([X[i, j] for i in range(self.M)]) <= 1)
        my_model.maximize(my_model.sum([self.match_mat[i][j] * X[i, j] for i in range(self.M) for j in range(self.N)]))
        sln = my_model.solve()
        result1 = [[int(sln[X[i, j]]) for i in range(self.M)] for j in range(self.N)]    # [N, M]
        obj = sln.get_objective_value() + sum([self.slab_data[i][0] * self.self_design for i in range(self.M)])
        return result1, obj





