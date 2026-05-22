# Sharedcam 与 Contactphase 数学公式说明

这份说明总结了当前篮球 shared-camera 分支里使用的优化目标。

## 1. 记号定义

对第 $t$ 帧：

- 相机坐标系下的球心平移：
  $$T_t = (X_t, Y_t, Z_t)$$
- 投影后的球心坐标：
  $$(u_t, v_t)$$
- 投影后的球半径（像素）：
  $$r_t$$
- 观测到的球心 / 半径：
  $$(u_t^{obs}, v_t^{obs}, r_t^{obs})$$
- 图像中的共享地面线：
  $$floor_v$$
- 投影后的球底像素位置：
  $$bottom_v(t)$$

篮球半径固定为：

$$
R = 0.12 \text{ m}
$$

投影模型为：

$$
u_t = f_x \frac{X_t}{Z_t} + c_x
$$

$$
v_t = f_y \frac{Y_t}{Z_t} + c_y
$$

$$
r_t = f_x \frac{R}{Z_t}
$$

$$
bottom_v(t) = v_t + r_t
$$

---

## 2. Sharedcam baseline

代码位置：
- `scripts/shared/sharedcam/run_basketball_pose6d_sharedcam.py`

当前 shared-camera baseline 的目标函数为：

$$
E_{sharedcam}
= \lambda_{mask} E_{mask}
+ \lambda_{center} E_{center}
+ \lambda_{size} E_{size}
+ \lambda_{contact} E_{contact}
+ \lambda_{temp} E_{temp}^{xyz}
+ \lambda_{ztemp} E_{temp}^{z}
+ \lambda_{zb} E_{z-boundary}
+ \lambda_{zs} E_{z-slope}
+ E_{aux}
$$

### 2.1 Mask / contour 项

让投影出来的圆轮廓尽量贴近观测 mask 的轮廓：

$$
E_{mask} = \sum_t d_{chamfer}(C_t^{proj}, C_t^{obs})
$$

当前权重：

$$
\lambda_{mask} = 0.018
$$

### 2.2 Center reprojection 项

$$
E_{center} = \sum_t \left[(u_t-u_t^{obs})^2 + (v_t-v_t^{obs})^2\right]
$$

当前权重：

$$
\lambda_{center} = 0.04
$$

### 2.3 Size 项

让投影后的直径 / 面积贴近观测 mask 的宽、高、平均尺度和面积：

$$
E_{size} = \sum_t \Big[(d_t-w_t)^2 + (d_t-h_t)^2 + (d_t-s_t)^2 + \alpha (a_t-a_t^{obs})^2\Big]
$$

其中 $d_t = 2r_t$。

当前权重：

$$
\lambda_{size} = 0.02
$$

### 2.4 Contact / floor 项

在接触帧上，让投影后的球底尽量贴近共享地面线：

$$
E_{contact} = \sum_{t \in C} (bottom_v(t)-floor_v)^2
$$

当前权重：

$$
\lambda_{contact} = 10.0
$$

### 2.5 三维时序平滑项

对平移使用二阶差分平滑：

$$
E_{temp}^{xyz} = \sum_t \|T_{t+1} - 2T_t + T_{t-1}\|^2
$$

当前权重：

$$
\lambda_{temp} = 0.08
$$

### 2.6 深度额外平滑项

$$
E_{temp}^{z} = \sum_t (Z_{t+1} - 2Z_t + Z_{t-1})^2
$$

当前权重：

$$
\lambda_{ztemp} = 0.22
$$

### 2.7 分段边界连续项

约束相邻分段之间的深度连续性：

$$
E_{z-boundary} = \sum_k (Z_{start(k+1)} - Z_{end(k)})^2
$$

当前权重：

$$
\lambda_{zb} = 3.5
$$

### 2.8 分段斜率连续项

$$
E_{z-slope} = \sum_k (a_{k+1} - a_k)^2
$$

当前权重：

$$
\lambda_{zs} = 0.35
$$

### 2.9 其他辅助正则项

当前实现里还有几项较小的辅助正则：

- 最小深度惩罚
- 分段斜率正则
- 首尾稳定项

对应的硬编码系数为：

$$
0.30, \; 0.12, \; 0.15
$$

---

## 3. Contactphase refinement

代码位置：
- `scripts/shared/human_ball/contact/run_human_ball_contact_phase_calibration.py`

这一阶段不会把整条球轨迹完全重解，而是保持观测到的 2D 球心不变，只在接触帧附近对深度轨迹做局部 refinement。

主要优化变量是：

$$
Z_t
$$

然后再由 $(u_t^{obs}, v_t^{obs}, Z_t)$ 反推出 $(X_t, Y_t)$。

目标函数为：

$$
E_{contactphase}
= w_{size} E_{size}
+ w_{fw} E_{floor-window}
+ w_{fc} E_{floor-contact}
+ w_{anchor} E_{anchor}
+ w_{temp} E_{temp}

\text{subject to } Z_t = Z_{hand}(t),\; t \in C
$$

### 3.1 Size 项

保持投影半径和观测半径一致：

$$
E_{size} = \sum_t (r_t-r_t^{obs})^2
$$

当前权重：

$$
w_{size} = 0.12
$$

### 3.2 Floor-window 项

在接触帧附近的小窗口里，约束球底不要偏离 floor 太多：

$$
E_{floor-window} = \sum_t w_t^{window} (bottom_v(t)-floor_v(t))^2
$$

当前权重：

$$
w_{fw} = 0.45
$$

当前窗口参数：

$$
radius = 3 \text{ frames}, \quad \sigma = 1.35
$$

### 3.3 Exact-contact floor 项

在 exact contact 帧上更强地约束球底贴近 floor：

$$
E_{floor-contact} = \sum_{t \in C} (bottom_v(t)-floor_v(t))^2
$$

当前权重：

$$
w_{fc} = 1.35
$$

### 3.4 Contact 帧上的手部 Z 硬约束

先用 SMPL-X joints 构造一个粗略的 palm proxy：

$$
P_t^{palm} = \frac{1}{5}\left(P_t^{wrist} + P_t^{index1} + P_t^{middle1} + P_t^{ring1} + P_t^{pinky1}\right)
$$

然后每一帧选择离当前球估计更近的那只手作为 active hand。

在 exact contact 帧上，直接强制球的深度等于 active hand 的深度：

$$
Z_t = Z_{hand}(t), \quad t \in C
$$

这里已经不是 soft penalty，而是硬约束。也就是说，在实现里会先把 contact 帧上的球 `Z` 直接替换成手的 `Z`，然后再去优化其余帧。

因此，这一版不再通过一个带权重的 hand-contact loss 去“鼓励接近”，而是把 contact 帧深度直接固定住。

### 3.5 Anchor 项

让 refinement 后的深度在非 contact 帧上不要离全局参考轨迹太远：

$$
E_{anchor} = \sum_{t \notin C} (Z_t - Z_t^{ref})^2
$$

当前权重：

$$
w_{anchor} = 0.70
$$

### 3.6 Temporal 项

对 refinement 后的深度轨迹做二阶平滑：

$$
E_{temp} = \sum_t (Z_{t+1} - 2Z_t + Z_{t-1})^2
$$

当前权重：

$$
w_{temp} = 1.35
$$

---

## 4. 高层理解

### Sharedcam

`sharedcam` baseline 的作用是：  
把球放进和 GVHMR human 一样的相机几何里，通过 reprojection、size、floor contact 和 temporal smoothness，得到一条比较稳定的球轨迹。

### Contactphase

`contactphase` 的作用是：  
在 contact 帧上直接把球的 `Z` 锁到手的 `Z`，然后只对其余帧做 refinement，同时尽量保持 floor consistency 和 temporal smoothness。
