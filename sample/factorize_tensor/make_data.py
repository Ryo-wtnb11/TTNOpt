import numpy as np
import matplotlib.pyplot as plt

np.random.seed(12345)

# パラメータ設定
L = 8  # x, y, z のビット数
n = 30 # 余弦項の数

# k_j ベクトルをランダムに生成
k_vectors = np.random.normal(0, 1, (n, 3))

# 量子状態ベクトルの初期化 (2^24 次元)
quantum_state_size = 2**(L * 3)
quantum_state_standard = np.zeros(quantum_state_size)
quantum_state_interleaved = np.zeros(quantum_state_size)

# バッチ処理の設定
batch_size = 2**20  # メモリ効率を考慮し、バッチごとに処理
num_batches = quantum_state_size // batch_size  # バッチ数

for batch in range(num_batches):
    start_index = batch * batch_size
    end_index = start_index + batch_size

    # インデックスをバイナリに変換し、各ビット列を分割
    indices = np.arange(start_index, end_index, dtype=np.uint32)
    binary_rep = ((indices[:, None] >> np.arange(L*3)[::-1]) & 1).astype(np.uint8)

    # 標準形式のビット列に基づく x, y, z
    x_bin_standard = binary_rep[:, :L]
    y_bin_standard = binary_rep[:, L:L*2]
    z_bin_standard = binary_rep[:, L*2:]

    # インターリーブ形式のビット列を構築
    interleaved_rep = np.empty((batch_size, L * 3), dtype=np.uint8)
    interleaved_rep[:, 0::3] = binary_rep[:, :L]  # x のビット
    interleaved_rep[:, 1::3] = binary_rep[:, L:L*2]  # y のビット
    interleaved_rep[:, 2::3] = binary_rep[:, L*2:]  # z のビット

    # インターリーブ形式から x, y, z を分割
    x_bin_interleaved = interleaved_rep[:, 0::3]
    y_bin_interleaved = interleaved_rep[:, 1::3]
    z_bin_interleaved = interleaved_rep[:, 2::3]

    # 各ビット列を用いて [0, 1] の範囲に正規化 (2進数から実数値への変換)
    x_vals_standard = np.sum(x_bin_standard / 2**(np.arange(1, L + 1)), axis=1)
    y_vals_standard = np.sum(y_bin_standard / 2**(np.arange(1, L + 1)), axis=1)
    z_vals_standard = np.sum(z_bin_standard / 2**(np.arange(1, L + 1)), axis=1)

    x_vals_interleaved = np.sum(x_bin_interleaved / 2**(np.arange(1, L + 1)), axis=1)
    y_vals_interleaved = np.sum(y_bin_interleaved / 2**(np.arange(1, L + 1)), axis=1)
    z_vals_interleaved = np.sum(z_bin_interleaved / 2**(np.arange(1, L + 1)), axis=1)

    # 量子状態ベクトルに関数 f(r) を適用して計算
    for j in range(1, n + 1):
        quantum_state_standard[start_index:end_index] += np.cos(j * (k_vectors[j-1, 0] * x_vals_standard + k_vectors[j-1, 1] * y_vals_standard + k_vectors[j-1, 2] * z_vals_standard))
        quantum_state_interleaved[start_index:end_index] += np.cos(j * (k_vectors[j-1, 0] * x_vals_interleaved + k_vectors[j-1, 1] * y_vals_interleaved + k_vectors[j-1, 2] * z_vals_interleaved))

# 量子状態を (2, 2, ..., 2) にreshape
quantum_state_standard = quantum_state_standard.reshape((2,) * (L*3))
quantum_state_interleaved = quantum_state_interleaved.reshape((2,) * (L*3))

# 計算した量子状態を保存
np.save("quantum_state_standard.npy", quantum_state_standard)
np.save("quantum_state_interleaved.npy", quantum_state_interleaved)

# z = 0.5 に対応するビット列を計算
z_fixed_bin = [1] + [0] * 7  # z0 = 1 で他は 0
z_vals_fixed = sum(z_fixed_bin[i] / 2**(i+1) for i in range(8))

# x, y の値を生成 (範囲 [0, 1])
grid_size = 2**8  # 8ビットの範囲で [0, 1]
x_vals_fixed = np.linspace(0, 1, grid_size)
y_vals_fixed = np.linspace(0, 1, grid_size)

# f(x, y, z=0.5) を計算する
f_xy_fixed_z = np.zeros((grid_size, grid_size))
for i, x in enumerate(x_vals_fixed):
    for j, y in enumerate(y_vals_fixed):
        f_xy_fixed_z[i, j] = sum(
            np.cos(k * (k_vectors[k-1, 0] * x + k_vectors[k-1, 1] * y + k_vectors[k-1, 2] * z_vals_fixed))
            for k in range(1, n + 1)
        )

# 結果を描画し、カラーバー範囲を -10 から 10 に設定
plt.figure(figsize=(6, 6))
plt.imshow(f_xy_fixed_z, extent=(0, 1, 0, 1), origin='lower', cmap='seismic', vmin=-10, vmax=10)
plt.colorbar(extend='both')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig("sample.pdf")
plt.close()
