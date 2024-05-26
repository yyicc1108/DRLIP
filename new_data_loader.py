# 柔性计算匹配费用 钢级相同 > 优充 > 以次充好


class SteelPlant:
    def __init__(self, filename):
        self.t = 0
        self.ord_data = list()
        self.s_data = list()
        self._load_data(filename)
        self.slab_yard = list()
        # print("slab_yard", self.slab_yard)
        self.order_yard = list()
        self.late_penalty = 1.0
        self.total_profit = 0.0

    def _load_data(self, filename):
        with open(filename[0], 'r') as f:
            for day, line in enumerate(f):
                data_ = []
                inp = line.strip()    # strip()：去除首尾空格  返回的是字符串inp outp
                #使用map，将inp中的
                data = list(map(float, inp.strip().split())) # 用map将分割好的字符串变成float,最后变成List
                for ord_num in range(len(data) // 3):
                    day_data = [day] + data[ord_num * 3: ord_num * 3 + 3]
                    data_.append(day_data)
                self.ord_data.append(data_)

        with open(filename[1], 'r') as f:
            for day, line in enumerate(f):
                data_ = []
                inp = line.strip()
                data = list(map(float, inp.strip().split())) # 用map将分割好的字符串变成float,最后变成List
                for slab_num in range(len(data) // 2):
                    day_data = [day] + data[slab_num * 2: slab_num * 2 + 2]
                    data_.append(day_data)
                self.s_data.append(data_)

    def get_data(self, days):
        temp_day = self.t
        # 把每一天的数据，从第一天到第days天，按顺序写入列表中
        for day in range(days):
            for s in self.s_data[self.t]:
                s[0] -= temp_day
                self.slab_yard.append(s)
            for o in self.ord_data[self.t]:
                o[0] -= temp_day
                self.order_yard.append(o)
            self.t += 1
        # print("slab_data:", self.slab_yard)
        # print("order_yard:", self.order_yard)
        return self.slab_yard.copy(), self.order_yard.copy()

    def return_data(self, ret_slab, ret_order, T):
        for ord_idx in range(len(ret_order)):
            ret_order[ord_idx][3] = int(ret_order[ord_idx][3] + ret_order[ord_idx][0] - T)
            ret_order[ord_idx][0] = 0
        self.slab_yard = ret_slab.copy()
        self.order_yard = ret_order.copy()



# loader = SteelPlant(("OrdData.txt", "slabData.txt"))
# slab_data, ord_data = loader.get_data(2)
# print(slab_data)
# print(ord_data)


