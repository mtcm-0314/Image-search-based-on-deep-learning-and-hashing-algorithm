def relative_unfairness(p1, p2, p3, n1, n2, n3):
    r1 = abs(p1 / n1 - p2 / n2)
    r2 = abs(p2 / n2 - p3 / n3)
    r3 = abs(p3 / n3 - p1 / n1)
    return max(r1, r2, r3)

def assign_seat(p1, p2, p3, n1, n2, n3):
    if p1 / (n1 + 1) > p2 / n2 and p1 / (n1 + 1) > p3 / n3:
        return 'A'
    elif p2 / (n2 + 1) > p1 / n1 and p2 / (n2 + 1) > p3 / n3:
        return 'B'
    elif p3 / (n3 + 1) > p1 / n1 and p3 / (n3 + 1) > p2 / n2:
        return 'C'
    else:
        r_a_increase = relative_unfairness(p1, p2, p3, n1 + 1, n2, n3)
        r_b_increase = relative_unfairness(p1, p2, p3, n1, n2 + 1, n3)
        r_c_increase = relative_unfairness(p1, p2, p3, n1, n2, n3 + 1)
        min_r_increase = min(r_a_increase, r_b_increase, r_c_increase)
        if min_r_increase == r_a_increase:
            return 'A'
        elif min_r_increase == r_b_increase:
            return 'B'
        else:
            return 'C'

# 示例
p1 = 103  # 甲方人数
p2 = 63   # 乙方人数
p3 = 34   # 丙方人数
n1 = 20   # 初始甲方席位数
n2 = 20   # 初始乙方席位数
n3 = 20   # 初始丙方席位数

seats_to_assign = 20  # 待分配的总席位数

while seats_to_assign > 0:
    assign_to = assign_seat(p1, p2, p3, n1, n2, n3)
    if assign_to == 'A':
        n1 += 1
    elif assign_to == 'B':
        n2 += 1
    else:
        n3 += 1
    seats_to_assign -= 1

print("分配结果：")
print("甲方分得席位数：", n1)
print("乙方分得席位数：", n2)
print("丙方分得席位数：", n3)
