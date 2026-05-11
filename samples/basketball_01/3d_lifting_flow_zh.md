# Basketball 3D Lifting Flow

这份说明只描述当前正在使用的 **basketball 3D lifting baseline**，不包含之前的 pseudo-3D 可视化分支。

## 1. 目标

从单目篮球视频中恢复一个最小可行的 3D 球心轨迹：

```text
video -> mask / 2D trajectory -> UVZ estimation -> 3D lifting -> scene rendering
```

最终目标是得到每一帧的：

```text
(X, Y, Z)
```

其中：

- `X`：横向位置
- `Y`：离地高度
- `Z`：相机前向深度

---

## 2. 当前输入文件

当前流程主要依赖下面几个文件：

### 2.1 `results/segmentation/masks/*.png`

SAM2 逐帧输出的篮球 mask。

作用：

- 定位篮球在图像中的区域
- 提取球心和表观半径

### 2.2 `results/tracking/ball_trajectory.csv`

当前版本的主轨迹来自 **mask circle fit**，而不是 CoTracker center。

主要字段：

- `frame`
- `time`
- `ball_center_x`
- `ball_center_y`
- `radius`
- `source = sam2_mask_circle`

其中：

- `ball_center_x -> u`
- `ball_center_y -> v`
- `radius -> r`

### 2.3 `results/events/visual_events.csv`

包含视觉上检测到的 bounce / contact frame。

作用：

- 估计地面接触时刻
- 给后续 `Y` 方向拟合提供约束

---

## 3. Step 1: 从 mask 提取 image-space 几何量

脚本位置：

- [scripts/shared/run_cotracker_basketball.py](/mnt/hdd/AudioHOI/scripts/shared/run_cotracker_basketball.py)

当前主轨迹不是直接用 mask centroid，而是对每帧 mask 做圆拟合：

```text
mask -> fitted circle -> (u, v, r)
```

这样每一帧得到：

- `u`: 球心图像横坐标
- `v`: 球心图像纵坐标
- `r`: 图像中的表观半径

这一步的目的是让后续 lifting 使用一组更自洽的球几何量。

---

## 4. Step 2: UVZ 预估逻辑

脚本位置：

- [scripts/shared/run_basketball_3d_lifting.py](/mnt/hdd/AudioHOI/scripts/shared/run_basketball_3d_lifting.py)

### 4.1 已知假设

当前 baseline 使用以下最小假设：

1. 篮球真实半径固定：

```text
R = 0.12 m
```

2. 相机近似 pinhole camera

3. 焦距近似：

```text
f_x = f_y = 0.9 * image_width
```

对于 1280 宽的视频：

```text
f ≈ 1152 px
```

4. 图像主点近似在中心：

```text
c_x = image_width / 2
c_y = image_height / 2
```

### 4.2 Z 的估计

利用球的真实半径固定、图像里表观半径可测这一点，当前深度估计公式是：

```text
Z_raw = f * R / r
```

含义：

- `r` 越大，球越近，`Z` 越小
- `r` 越小，球越远，`Z` 越大

这一步就是当前流程里的 **monocular size-based depth estimation**。

### 4.3 X 的估计

有了 `Z` 之后，用图像横向位置反投影到 3D：

```text
X_raw = (u - c_x) * Z_raw / f_x
```

含义：

- 如果球在图像中偏左，`X` 为负
- 如果球在图像中偏右，`X` 为正

### 4.4 Y 的估计

`Y` 不是直接由 `v` 投影得到，而是结合“球底到地面”的像素差来估计。

先计算球底：

```text
bottom_v = v + r
```

再从 visual contact frames 中估计一个地面像素位置：

```text
floor_v = median(bottom_v at contact frames)
```

然后把球底离地面的像素差转成高度：

```text
Y_raw = R + max(0, floor_v - bottom_v) * Z_raw / f_y
```

含义：

- 接地时，球心高度应接近球半径 `R`
- 球底越高于 floor，球心高度越大

所以当前的 `Y` 是：

```text
ground-anchored height estimate
```

而不是简单的 image-space `v` 直接投影。

---

## 5. Step 3: 3D 轨迹重建与拟合

当前脚本不会直接把 `X_raw, Y_raw, Z_raw` 作为最终结果，而是再做一层分段拟合。

### 5.1 为什么要拟合

原始逐帧估计会受到以下因素影响：

- mask 抖动
- 半径估计误差
- 接触帧局部噪声

所以需要把轨迹变成更平滑、也更符合运动规律的 3D 路径。

### 5.2 拟合策略

以 bounce/contact frame 为边界，把轨迹分段。

#### X 拟合

每个 segment 内：

```text
X -> 线性拟合
```

#### Z 拟合

每个 segment 内：

```text
Z -> 线性拟合
```

#### Y 拟合

每个 segment 内：

```text
Y -> 二次曲线拟合
```

同时在 contact frame 上强制：

```text
Y = R
```

这表示球在接地帧时，球心高度应等于球半径。

因此当前 `Y` 拟合本质上是：

```text
contact-constrained parabolic fitting
```

---

## 6. 当前输出文件

### 6.1 `results/lifting/ball_3d_lifted_trajectory.csv`

这是当前 3D lifting 的主输出。

主要字段：

- `u, v, radius_px`
- `X_raw, Y_raw, Z_raw`
- `X, Y, Z`

其中：

- `raw` 表示直接从公式得到的逐帧估计
- `X, Y, Z` 表示拟合和平滑后的最终轨迹

### 6.2 `results/ball_3d_lifted_plot.png`

标准 3D 轨迹图。

### 6.3 `results/ball_3d_lifted_components.png`

展示 `X/Y/Z` 的 raw 与 fit 对比。

### 6.4 `results/lifting/ball_3d_reprojection_comparison.csv/png`

把最终 3D 轨迹重新投回 2D，和原始 `u/v` 做比较。

作用：

- 检查 lifting 是否自洽
- 判断当前 3D baseline 是否基本合理

### 6.5 `results/renders/lifted_scene*.mp4/png`

基于 `ball_3d_lifted_trajectory.csv` 的 3D scene render。

---

## 7. 当前 `render_lifted_scene.py` 的逻辑

脚本位置：

- [scripts/shared/render_lifted_scene.py](/mnt/hdd/AudioHOI/scripts/shared/render_lifted_scene.py)

这个脚本现在只吃：

- `results/lifting/ball_3d_lifted_trajectory.csv`

它不再依赖之前的 pseudo-3D UV trick。

当前支持两种视图：

### 7.1 `world` 视图

直接把 `X/Y/Z` 画成独立 3D 场景：

- 地面平面
- 完整历史轨迹
- 当前球位置
- 球在地面的 shadow

### 7.2 `camera` 视图

把同样的 `X/Y/Z` 放到一个固定 synthetic camera 下，再投影回 2D：

- 使用固定相机高度、俯仰角、焦距
- 画出更像原视频视角的 3D scene

这一步是：

```text
XYZ-based camera render
```

而不是之前那种基于 image-space 直接画出来的 pseudo-3D。

---

## 8. 当前全流程总结

可以把当前篮球 baseline 压成这 6 步：

### Step A

```text
video -> SAM2 masks
```

### Step B

```text
mask -> fitted circle -> (u, v, r)
```

### Step C

```text
r -> Z_raw
u + Z_raw -> X_raw
bottom_v + floor_v + Z_raw -> Y_raw
```

### Step D

```text
bounce/contact frames -> segment boundaries
```

### Step E

```text
X_raw, Y_raw, Z_raw -> fitted X, Y, Z
```

### Step F

```text
ball_3d_lifted_trajectory.csv -> 3D plots / lifted scene render
```

---

## 9. 当前方法的性质

这条线的本质是：

```text
monocular 3D lifting baseline
```

不是：

- 完整 4D reconstruction
- 多视角几何重建
- 学习式 end-to-end 3D recovery

它的定位是：

```text
用最小几何假设，从 2D mask trajectory lift 到一个可分析、可渲染、可继续改进的 3D ball trajectory baseline
```

---

## 10. 当前已知局限

1. `Z` 强依赖表观半径 `r`，对 mask 误差敏感
2. `Y` 强依赖 ground/contact 假设
3. 焦距 `f` 目前是近似值，不是严格标定值
4. 结果是 baseline，不是精确物理真值

但它已经把当前流程从：

```text
pseudo-3D visualization
```

推进到了：

```text
explicit UVZ estimation + geometric 3D lifting + 3D scene rendering
```
