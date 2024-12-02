import numpy as np
import pandas as pd


def open_adjacent_indexs(d: int):
    n = 2**d
    ind_list = [[i, (i + 1)] for i in range(n - 1)]
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
    Jxx_list = [-4.0 for _ in ij_list]
    Jyy_list = [0.0 for _ in ij_list]
    Jzz_list = [0.0 for _ in ij_list]
    df = pd.DataFrame(np.array([Jxx_list, Jyy_list, Jzz_list]).T)
    df = pd.concat([df_ij, df], axis=1)
    df.to_csv("coupling.csv", index=False, header=False)
