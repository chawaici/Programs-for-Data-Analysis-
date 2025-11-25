import numpy as np
import matplotlib.pyplot as plt

def find_peak_near(x, y, center, window=20):
    """在给定中心附近找最高峰"""
    mask = (x >= center - window) & (x <= center + window)
    if not np.any(mask):
        return None, None
    x_sub = x[mask]
    y_sub = y[mask]
    idx = np.argmax(y_sub)
    return x_sub[idx], y_sub[idx]

def main():
    file_path = "analysis.txt"

    # 读取数据：每行最后两列为 x、y
    x_vals, y_vals = [], []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                x = float(parts[-2])
                y = float(parts[-1])
            except ValueError:
                continue
            x_vals.append(x)
            y_vals.append(y)

    x = np.array(x_vals)
    y = np.array(y_vals)

    # 只保留 300–1000 nm 范围内的数据
    mask_roi = (x >= 300) & (x <= 1000)
    x = x[mask_roi]
    y = y[mask_roi]

    fig, ax = plt.subplots()

    # 折线图（红色，平滑后的曲线）
    ax.plot(x, y, color="red")

    # 四周框线
    for spine in ax.spines.values():
        spine.set_visible(True)

    # x 轴：300~1000，每 100 一个刻度，刻度线向下
    ax.set_xlim(300, 1000)
    xticks = np.arange(300, 1000 + 1, 100)
    ax.set_xticks(xticks)
    ax.tick_params(axis="x", direction="out")
    for label in ax.get_xticklabels():
        label.set_fontname("Times New Roman")

    # 轴标签
    ax.set_xlabel("wavelength(nm)", fontname="Times New Roman")
    ax.set_ylabel("Intensity", fontname="Times New Roman")

    # y 轴不带刻度
    ax.yaxis.set_ticks([])
    ax.tick_params(axis="y", length=0)

    # ---- 在 ~500 nm 和 ~800 nm 处标记峰值 ----
    # 使用平滑后的数据找峰
    x_peak1, y_peak1 = find_peak_near(x, y, center=500, window=30)
    x_peak2, y_peak2 = find_peak_near(x, y, center=780, window=50)

    # 画深蓝色标记与文字
    if x_peak1 is not None:
        ax.scatter([x_peak1], [y_peak1], color="navy", s=25, zorder=3)
        ax.text(x_peak1, y_peak1,
                f"{x_peak1:.1f} nm",
                color="navy",
                fontsize=8,
                ha="left", va="bottom",
                fontname="Times New Roman")

    if x_peak2 is not None:
        ax.scatter([x_peak2], [y_peak2], color="navy", s=25, zorder=3)
        ax.text(x_peak2, y_peak2,
                f"{x_peak2:.1f} nm",
                color="navy",
                fontsize=8,
                ha="left", va="bottom",
                fontname="Times New Roman")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
