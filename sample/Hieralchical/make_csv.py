import numpy as np
import pandas as pd


def open_adjacent_indexs(d: int):
    n = 2**d
    if n > 2:
        ind_list = [[i, (i + 1)] for i in range(n - 1)]
    else:
        ind_list = [[0, 1]]
    return ind_list


def generate_height_list(d: int):
    def assign_heights(start, end, height, heights):
        if start > end:
            return
        mid = (start + end) // 2
        heights[mid] = height
        assign_heights(start, mid - 1, height - 1, heights)
        assign_heights(mid + 1, end, height - 1, heights)

    length = 2**d - 1
    heights = [0] * length
    assign_heights(0, length - 1, d - 1, heights)
    return heights


if __name__ == "__main__":
    d = 8
    size = 2**d

    ij_list = open_adjacent_indexs(d)
    df_ij = pd.DataFrame(np.array(ij_list), columns=["i", "j"])
    coefs = generate_height_list(d)
    alpha = 1.0
    coefs = [1.0 * (alpha**coef) for coef in coefs]
    interaction_coefs = [[coef, coef] for coef in coefs]
    df = pd.DataFrame(np.array(interaction_coefs), columns=["XY", "Z"])
    df = pd.concat([df_ij, df], axis=1)
    df.to_csv(f"a={alpha}/xxz.csv", index=False, header=False)
