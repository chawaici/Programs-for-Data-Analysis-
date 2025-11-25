import numpy as np
import matplotlib.pyplot as plt

def find_peak_near(x, y, center, window=20):
    mask = (x >= center - window) & (x <= center + window)
    if not np.any(mask):
        return None, None

    x_sub = x[mask]
    y_sub = y[mask]

    idx_max = np.argmax(y_sub)
    x_peak = x_sub[idx_max]
    y_peak = y_sub[idx_max]

    baseline = np.min(y_sub)
    half_val = baseline + 0.5 * (y_peak - baseline)

    # 左侧半高点
    i_left = idx_max
    while i_left > 0 and y_sub[i_left] > half_val:
        i_left -= 1
    if i_left == idx_max:
        x_left = x_peak
    else:
        x1, x2 = x_sub[i_left], x_sub[i_left + 1]
        y1, y2 = y_sub[i_left], y_sub[i_left + 1]
        x_left = x1 + (half_val - y1) * (x2 - x1) / (y2 - y1) if y2 != y1 else x1

    # 右侧半高点
    i_right = idx_max
    n = len(y_sub)
    while i_right < n - 1 and y_sub[i_right] > half_val:
        i_right += 1
    if i_right == idx_max:
        x_right = x_peak
    else:
        x1, x2 = x_sub[i_right - 1], x_sub[i_right]
        y1, y2 = y_sub[i_right - 1], y_sub[i_right]
        x_right = x1 + (half_val - y1) * (x2 - x1) / (y2 - y1) if y2 != y1 else x2

    return 0.5 * (x_left + x_right), half_val

def main():
    file_path = "analysis.txt"  # 换成你的真实文件名称即可

    x_vals, y_vals = [], []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            try:
                x_vals.append(float(parts[-2]))
                y_vals.append(float(parts[-1]))
            except:
                continue

    x = np.array(x_vals)
    y = np.array(y_vals)

    mask = (x >= 300) & (x <= 1000)
    x = x[mask]
    y = y[mask]

    fig, ax = plt.subplots()

    ax.plot(x, y, color="red")

    for spine in ax.spines.values():
        spine.set_visible(True)

    ax.set_xlim(300, 1000)
    ax.set_xticks(np.arange(300, 1001, 100))
    for tick in ax.get_xticklabels():
        tick.set_fontname("Times New Roman")
        tick.set_fontsize(14)
        tick.set_fontweight("bold")

    ax.tick_params(axis="x", direction="out")
    for t in ax.get_xticklabels():
        t.set_fontname("Times New Roman")

    ax.set_xlabel(
        "Wavelength(nm)",      # 斜体
        fontname="Times New Roman",
        fontweight="bold",                  # 加粗
        fontsize=18                        # 比原来大两个字号
    )

    ax.set_ylabel(
        "Intensity",
        fontname="Times New Roman",
        fontweight="bold",                  # 加粗
        fontsize=18
    )

    ax.yaxis.set_ticks([])
    ax.tick_params(axis="y", length=0)

    ymin, ymax = ax.get_ylim()

    # 找两处半高宽中点
    x_peak1, y_peak1 = find_peak_near(x, y, center=500, window=30)
    x_peak2, y_peak2 = find_peak_near(x, y, center=775, window=50)

    # 画贯穿底部到顶部的虚线（蓝色）
    def draw_vertical(xp):
        ax.vlines(xp, ymin, ymax,
                  colors="blue",
                  linestyles="dashed",
                  linewidth=1.2)

        # 标注文字在虚线右侧、靠近顶边框
        ax.text(xp + 5,
                ymax - 0.02 * (ymax - ymin),
                f"{xp:.1f} nm",
                color="blue",
                fontsize=11,
                ha="left",
                va="top",
                fontname="Times New Roman")

    if x_peak1 is not None:
        draw_vertical(x_peak1)
    if x_peak2 is not None:
        draw_vertical(x_peak2)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
