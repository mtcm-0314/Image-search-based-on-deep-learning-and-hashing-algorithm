import math

def d(n):
    return math.sqrt(n * (n + 1))

def assign_seat(p, s):
    # 初始化每个方分配的席位数
    f = [0] * len(p)

    # 从席位数 s 递增到目标总席位数 N
    for i in range(s, N):
        # 计算每个方的权重
        weights = [p[j] / d(f[j] + 1) for j in range(len(p))]
        # 找出权重最大的方
        max_weight_index = weights.index(max(weights))
        # 分配席位给权重最大的方
        f[max_weight_index] += 1

    return f

# 示例
p = [103, 63, 34]  # 甲乙丙三方人数
N = 20             # 总席位数

seat_distribution = assign_seat(p, N)

print("分配结果：")
for i in range(len(p)):
    print(f"第 {i+1} 方分得的席位：{seat_distribution[i]}")
