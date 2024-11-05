import new_data_loader
import docplex.mp.linear as linear
from docplex.mp.model import Model
import numpy as np
import time
import pandas as pd

obj_list_11 = list()
obj_list_12 = list()
obj_list_2 = list()
obj_list_3 = list()
best_fit = list()

EPSILON = 1.1

class Scenario:
    def __init__(self, new_slabs=None, new_orders=None, scenario_=None, id_=None):
        if scenario_ is None:
            self.my_model = Model("scenario tree")
            self.t = 0
            self.id = 0
            self.slabs = new_slabs[:]
            self.orders = new_orders[:]
            self.M = len(self.slabs)
            self.N = len(self.orders)
            self.X1 = self.my_model.binary_var_matrix(keys1=[i for i in range(self.M)],
                                                    keys2=[i for i in range(self.N)],
                                                    name='X1_0_0')
            self.X2 = self.my_model.binary_var_list(keys=[i for i in range(self.M)],
                                                    name='X2_0_0')
        else:
            self.my_model = scenario_.my_model
            self.t = scenario_.t + 1
            self.id = id_
            self.slabs = scenario_.slabs[:]
            self.orders = scenario_.orders[:]
            # sample new data
            self.slabs += new_slabs
            self.orders += new_orders
            # init vars
            self.M = len(self.slabs)
            self.N = len(self.orders)
            self.X1 = self.my_model.binary_var_matrix(keys1=[i for i in range(self.M)],
                                            keys2=[i for i in range(self.N)],
                                            name='X1_' + str(self.t) + '_' + str(self.id))
            self.X2 = self.my_model.binary_var_list(keys=[i for i in range(self.M)],
                                                    name='X2_' + str(self.t) + '_' + str(self.id))
            for rowId in range(scenario_.M):
                for colId in range(scenario_.N):
                    self.X1[rowId, colId] = scenario_.X1[rowId, colId]
            for rowId in range(scenario_.M):
                self.X2[rowId] = scenario_.X2[rowId]


def generate_tree(span, slabs_, orders, sampler_):
    slab, order = slabs_, orders
    s_prev = [Scenario(slab, order)]
    s_tmp = list()
    for t in range(span):
        for s in s_prev:
            # left branch
            slab_, order_ = sampler_.get_data(1, t + 1)
            sampler_.return_data([], [], 1)
            s_tmp.append(Scenario(slab_, order_, s, len(s_tmp)))
            # right branch
            slab__, order__ = sampler_.get_data(1, t + 1)
            sampler_.return_data([], [], 1)
            s_tmp.append(Scenario(slab__, order__, s, len(s_tmp)))
        s_prev = s_tmp.copy()
        s_tmp.clear()
    return s_prev

def build_sub_model(scenario, T):
    slab_data = scenario.slabs[:]
    order_data = scenario.orders[:]
    slabs_return = list()
    prev_inv_cost = 0.0

    M = len(slab_data)  # 板坯总个数
    N = len(order_data)  # 合同总个数
    # print(M, N)
    # 模型参数
    match_pro1 = 35.0  # 钢级相等
    match_pro2 = 1.5  # 板坯钢级大于合同钢级
    match_pro3 = 0.2  # 板坯钢级小于合同钢级
    sell_pro = 0.1  # 自拟合同
    unit_inventory_cost = 2.0  # 库存费用
    # variables
    my_model = scenario.my_model
    X1 = scenario.X1
    X2 = scenario.X2
    # # computing model coefficients:
    # 三个目标的权重
    f1 = 1.0  # 匹配利润
    f2 = 1.0  # 重量惩罚
    f3 = 2.0  # 欠量惩罚
    # 板坯到达时间(某天凌晨)，合同到达时间(某天凌晨)，合同交货时间（某天凌晨）
    # 例如：合同到达时间为1，生产时长为3，交货时间为3+1=4，合同在1，2，3这三天可以被匹配板坯，第四天一到就立即交货
    slab_arr_date = list()
    for i in range(M):
        slab_arr_date.append(slab_data[i][0])
    # print(slab_arr_date)
    order_arr_date = list()
    for i in range(N):
        order_arr_date.append(order_data[i][0])
    order_complete_date = list()
    for i in range(len(order_data)):
        order_complete_date.append(order_data[i][0] + int(order_data[i][3]))
    # 计算模型目标相关参数
    # 用户合同匹配费用： shape:[M, N]
    c1 = list()
    for i in range(M):
        c1_order_i = list()
        for j in range(N):
            if order_data[j][2] == slab_data[i][2]:
                c1_order_i.append(slab_data[i][1] * match_pro1)
            elif order_data[j][2] < slab_data[i][2]:
                c1_order_i.append(slab_data[i][1] * match_pro2)
            else:
                c1_order_i.append(slab_data[i][1] * match_pro3)
        c1.append(c1_order_i)
    # show_2d_value(c1)
    # 自拟合同匹配费用：
    c2 = list()
    for i in range(M):
        c2.append(slab_data[i][1] * sell_pro)
    # 板坯重量列表，合同容量列表：
    slab_weight = list()
    for i in range(M):
        slab_weight.append(slab_data[i][1])
    order_capacity = list()
    for i in range(N):
        order_capacity.append(order_data[i][1])
    # 匹配关系直接可以计算出库存费用：
    inv_cost = list()
    for i in range(M):
        inv_slab_order = list()
        for j in range(N):
            if order_arr_date[j] > slab_arr_date[i]:
                # 即使合同交货期>T，也要计算全部库存费用
                inv_slab_order.append(slab_weight[i] * (order_arr_date[j] - slab_arr_date[i]) * unit_inventory_cost)
            elif order_arr_date[j] <= slab_arr_date[i] <= order_complete_date[j]:
                inv_slab_order.append(0)
            else:
                inv_slab_order.append(9999)
        inv_cost.append(inv_slab_order)
    # 目标函数建模
    # 匹配利润
    obj1_1_list = list()
    for j in range(N):
        for i in range(M):
            if slab_arr_date[i] >= order_complete_date[j]:
                continue
            obj1_1_list.append(X1[i, j] * c1[i][j])
    # print(my_model.sum(obj1_1_list))
    obj1_2_list = list()
    for i in range(M):
        obj1_2_list.append(X2[i] * c2[i])
    obj1_1 = my_model.sum(obj1_1_list)
    obj1_2 = my_model.sum(obj1_2_list)
    obj1 = obj1_1 + obj1_2
    # 库存费用
    obj2_list = list()
    for i in range(M):
        slab_order_t_list = list()
        for j in range(N):
            if slab_arr_date[i] >= order_complete_date[j]:
                continue
            slab_order_t_list.append(X1[i, j])
            obj2_list.append(X1[i, j] * inv_cost[i][j])
        obj2_list.append((1 - my_model.sum(slab_order_t_list) - X2[i]) * unit_inventory_cost * slab_weight[i])
    obj2 = my_model.sum(obj2_list)
    # 合同完整性
    obj3_list = list()
    for j in range(N):
        if order_complete_date[j] > T:
            # 下一个时间段交货的合同，在下一时间段交货时计算完整性，
            # 第一天交货除外。因为到了第一天立即交货，下一时间段不会有板坯配给这样的合同
            continue
        slabs_order_j = list()
        for i in range(M):
            if slab_arr_date[i] >= order_complete_date[j]:
                continue
            slabs_order_j.append(-slab_weight[i] * X1[i, j])
        obj3_list.append(order_capacity[j] + my_model.sum(slabs_order_j))
    obj3 = my_model.sum(obj3_list)
    # 总目标函数
    obj = f1 * obj1 - f2 * obj2 - f3 * obj3
    ori_obj = my_model.get_objective_expr()
    my_model.maximize(obj + ori_obj)
    # # 添加约束
    # 变量关系约束
    for i in range(M):
        slab_order_t_list = list()
        for j in range(N):
            if order_complete_date[j] <= slab_arr_date[i]:
                continue
            slab_order_t_list.append(X1[i, j])
        my_model.add_constraint(my_model.sum(slab_order_t_list) + X2[i] <= 1)
    # 合同容量约束
    for j in range(N):
        slab_order_weight_list = list()
        for i in range(M):
            if slab_arr_date[i] >= order_complete_date[j]:
                continue
            slab_order_weight_list.append(slab_weight[i] * X1[i, j])
        my_model.add_constraint(order_capacity[j] - my_model.sum(slab_order_weight_list) >= 0)
    # set variable to zero for unacceptable pairs
    for j in range(N):
        for i in range(M):
            if slab_arr_date[i] >= order_complete_date[j]:
                X1[i, j].set_ub(0)

def solve_one_day(loader_, sampler_, span):
    slabs, orders = loader_.get_data(1, 0)
    # build scenario tree
    scenarios = generate_tree(span, slabs, orders, sampler_)
    # build IP model: add all constraints from all nodes of the tree to the model
    for scenario in scenarios:
        build_sub_model(scenario, span)
    # the the model from any of the leaf nodes and solve
    scenario_model = scenarios[0].my_model
    sln = scenario_model.solve()
    obj_val = scenario_model.get_objective_expr().solution_value
    X1 = scenarios[0].X1
    X2 = scenarios[0].X2
    # get the solution corresponding to the current day, regardless of the following days
    M = len(slabs)  # number of unassigned slabs of the current day
    N = len(orders) # number of unassigned orders of the current day
    result1 = [[round(sln[X1[i, j]]) for i in range(M)] for j in range(N)]
    result2 = [round(sln[X2[i]]) for i in range(M)]
    # # compute the profit
    # compute the match cost, completeness cost and inventory cost
    match_pro1 = 30.0  # 钢级相等
    match_pro2 = 1.5  # 板坯钢级大于合同钢级
    match_pro3 = 0.2  # 板坯钢级小于合同钢级
    sell_pro = 0.1  # 自拟合同
    unit_inventory_cost = 2.0  # 库存费用
    c1 = list()
    for i in range(M):
        c1_order_i = list()
        for j in range(N):
            if int(orders[j][2]) == int(slabs[i][2]):
                c1_order_i.append(slabs[i][1] * match_pro1)
            elif orders[j][2] < slabs[i][2]:
                c1_order_i.append(slabs[i][1] * match_pro2)
            else:
                c1_order_i.append(slabs[i][1] * match_pro3)
        c1.append(c1_order_i)
    c2 = list()
    for i in range(M):
        c2.append(slabs[i][1] * sell_pro)
    # compute the sum of all profits
    profits_sum = 0
    r1, r2, r3, r4 = 0, 0, 0, 0,
    for j in range(N):
        for i in range(M):      # assignment profit c1
            profits_sum += result1[j][i] * c1[i][j]
            r1 += result1[j][i] * c1[i][j]

        if orders[j][3] == 1:   # weight completeness
            profits_sum -= 1 * (orders[j][1] - sum([slabs[i][1] for i in range(M) if result1[j][i] == 1]))
            r3 += 1 * (orders[j][1] - sum([slabs[i][1] for i in range(M) if result1[j][i] == 1]))
    for i in range(M):          # assignment profit c2
        profits_sum += c2[i] * result2[i]
        r2 += c2[i] * result2[i]
    for i in range(M):          # inventory cost
        profits_sum -= (1 - sum([result1[j][i] for j in range(N)]) - result2[i])\
                       * unit_inventory_cost * slabs[i][1]

        r4 += (1 - sum([result1[j][i] for j in range(N)]) - result2[i])\
                       * unit_inventory_cost * slabs[i][1]
    # get all unassigned slabs and orders
    slabs_copy = slabs.copy()
    slabs_return = list()
    for i in range(M):
        if sum([result1[j][i] for j in range(N)]) + result2[i] == 0:
            slabs_return.append(slabs_copy[i])
    orders_copy = orders.copy()
    orders_return = list()
    for j in range(N):
        if orders_copy[j][0] + int(orders_copy[j][3]) > 1:  # 下一时间段第一天交货的合同不包含在内
            # 合同需求量要减去已经配上的板坯
            orders_copy[j][1] -= sum([slabs[i][1] * result1[j][i] for i in range(M)])
            orders_return.append(orders_copy[j])
    loader_.return_data(slabs_return, orders_return, 1)
    return profits_sum, r1, r2, r3, r4


DAYS = 30
SPAN = 3




sampler = new_data_loader.SteelPlant(("order-test-data.txt", "slab-test-data.txt"))
# load from the 100-th data, as sampled data
sampler.get_data(10)
sampler.return_data([], [], 0)
loader = new_data_loader.SteelPlant(("order-test-data.txt", "slab-test-data.txt"))

profit_sum = 0
profits = []
for day in range(DAYS):
    daily_profit, r1, r2, r3, r4 = solve_one_day(loader, sampler, SPAN)
    print(daily_profit, r1, r2, r3, r4)
    profits.append(daily_profit)
    profit_sum += daily_profit
print(profit_sum)
print(profits)
# slabs, orders = loader.get_data(1, 0)
# scenarios = generate_tree(2, slabs, orders, sampler_=sampler)
# show the tree
# for idx, s in enumerate(scenarios):
#     print(idx)
#     print(s.orders)
#     print(s.M, s.N)
#     for x in s.X2:
#         print(x.name)


