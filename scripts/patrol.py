import matplotlib
matplotlib.use('TkAgg')      # 👈 指定可交互后端
matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False

import numpy as np
import matplotlib.pyplot as plt
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# ---------- 省略 solve_tsp_fixed_end 与 plot_and_save_route ----------
def solve_tsp_fixed_end(points, start_index, end_index):
    """points: (N, 2) ndarray  |  start_index / end_index: 节点下标"""
    n = len(points)

    def distance(p1, p2):
        return int(np.linalg.norm(p1 - p2) * 1000)

    # 距离矩阵
    dist_matrix = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            if i != j:
                dist_matrix[i, j] = distance(points[i], points[j])

    # ✨ 关键行：起终点要用列表包裹
    manager = pywrapcp.RoutingIndexManager(n, 1, [start_index], [end_index])
    routing = pywrapcp.RoutingModel(manager)

    transit_cb_idx = routing.RegisterTransitCallback(
        lambda i, j: dist_matrix[manager.IndexToNode(i)][manager.IndexToNode(j)]
    )
    routing.SetArcCostEvaluatorOfAllVehicles(transit_cb_idx)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )

    solution = routing.SolveWithParameters(params)
    if not solution:
        raise RuntimeError("TSP 求解失败：无可行路径")

    # 解析路径
    idx = routing.Start(0)
    route = []
    while not routing.IsEnd(idx):
        route.append(manager.IndexToNode(idx))
        idx = solution.Value(routing.NextVar(idx))
    route.append(manager.IndexToNode(idx))
    return route

# ----------- 可视化并点击选点 -----------
def interactive_select_start_end(points_array):
    fig, ax = plt.subplots(figsize=(10, 10))
    scatter = ax.scatter(points_array[:, 1], points_array[:, 2], s=80, edgecolors='k')
    for i, (_, x, y, _) in enumerate(points_array):
        ax.text(x + 1, y + 1, str(i), fontsize=9)

    selected = []

    def onclick(event):
        if event.inaxes != ax:
            return
        x, y = event.xdata, event.ydata
        dists = np.linalg.norm(points_array[:, 1:3] - np.array([x, y]), axis=1)
        closest_index = np.argmin(dists)
        selected.append(closest_index)
        if len(selected) == 1:
            ax.scatter(points_array[closest_index, 1], points_array[closest_index, 2],
                       color='green', s=150, label='Start', edgecolors='black')
        elif len(selected) == 2:
            ax.scatter(points_array[closest_index, 1], points_array[closest_index, 2],
                       color='red', s=150, label='End', edgecolors='black')
            fig.canvas.mpl_disconnect(cid)
            plt.close(fig)

        fig.canvas.draw()

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.title("点击选择起点（绿）和终点（红）")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    ax.legend()
    plt.show()

    if len(selected) < 2:
        raise Exception("未选择起点和终点")

    return selected[0], selected[1]

# ----------- 路径可视化与保存 -----------
def plot_and_save_route(points_array, route, filename="smart_patrol_path.txt"):
    plt.figure(figsize=(10, 10))

    # 绘制路径线
    for i in range(len(route) - 1):
        x1, y1 = points_array[route[i], 1:3]
        x2, y2 = points_array[route[i + 1], 1:3]
        plt.plot([x1, x2], [y1, y2], 'b--')

    # 绘制点
    for i, (cls, x, y, conf) in enumerate(points_array):
        plt.scatter(x, y, s=conf * 80 + 30, edgecolors='k')
        plt.text(x + 1, y + 1, f"{i}", fontsize=8)

    sx, sy = points_array[route[0], 1:3]
    ex, ey = points_array[route[-1], 1:3]
    plt.scatter(sx, sy, color='green', s=150, edgecolors='black', label='Start')
    plt.scatter(ex, ey, color='red', s=150, edgecolors='black', label='End')

    plt.title("巡检最短路径")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.legend()
    plt.show()

    # 保存路径信息
    with open(filename, 'w') as f:
        f.write("index,cls,x,y,confidence\n")
        for i in route:
            cls, x, y, conf = points_array[i]
            f.write(f"{i},{int(cls)},{x:.2f},{y:.2f},{conf:.4f}\n")
    print(f"路径已保存到 {filename}")



def interactive_select_start_end(points_array):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(points_array[:, 1], points_array[:, 2], s=80, edgecolors='k')
    for i, (_, x, y, _) in enumerate(points_array):
        ax.text(x + 1, y + 1, str(i), fontsize=9)

    selected = []

    def onclick(event):
        if event.inaxes != ax or len(selected) >= 2:
            return
        x, y = event.xdata, event.ydata
        dists = np.linalg.norm(points_array[:, 1:3] - np.array([x, y]), axis=1)
        closest = int(np.argmin(dists))
        selected.append(closest)

        if len(selected) == 1:
            ax.scatter(*points_array[closest, 1:3], color='green',
                       s=150, label='Start', edgecolors='black')
        else:
            ax.scatter(*points_array[closest, 1:3], color='red',
                       s=150, label='End', edgecolors='black')

        # 刷新图例（去重）
        h, l = ax.get_legend_handles_labels()
        by_label = dict(zip(l, h))
        ax.legend(by_label.values(), by_label.keys())

        if len(selected) == 2:
            fig.canvas.mpl_disconnect(cid)   # 解绑事件
            plt.close(fig)                   # 关闭窗口
        else:
            fig.canvas.draw()

    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    plt.title("点击选择起点（绿）和终点（红）")
    plt.xlabel("X");  plt.ylabel("Y");  plt.grid(True)
    plt.show()

    if len(selected) < 2:
        raise Exception("未选择起点和终点")
    return selected[0], selected[1]

# ------------ 主程序调用 ------------
if __name__ == "__main__":
    points_array = np.load("merged_points.npy") # 你的点
    s, e = interactive_select_start_end(points_array)
    route = solve_tsp_fixed_end(points_array[:, 1:3], s, e)
    plot_and_save_route(points_array, route, "my_path.txt")