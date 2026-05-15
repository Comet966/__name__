# 📐 CameraSim × Depth-Mask-Engine 接口文档

> 景深识别引擎与 CameraSim 集成接口规范
> 最后更新: 2026-05-15

---

## 一、快速接入

在 CameraSim 里一行代码启用真实景深虚化：

```python
from depth_mask_engine import CameraSimDepthBridge

# 替换 image_processor.py 中原有的 simulate_aperture_blur()
img = CameraSimDepthBridge.apply_bokeh(img, aperture=2.8)
```

完整替换 `process()`：

```python
def process(bgr_img, aperture, ev_diff, iso):
    img = CameraSimDepthBridge.apply_bokeh(bgr_img, aperture)
    img = simulate_exposure(img, ev_diff)
    img = simulate_noise(img, iso)
    return img
```

---

## 二、核心 API

### `depth_mask_engine` 包

导入方式：
```python
from depth_mask_engine import (
    depth_mask,              # 路径 → 文件 （CLI 用）
    depth_mask_from_array,   # numpy → numpy （推荐）
    get_depth_map,           # 路径/数组 → 深度图
    CameraSimDepthBridge,    # 桥接类（推荐 CameraSim 使用）
)
```

### CameraSimDepthBridge（推荐）

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `compute_depth_map(image)` | BGR (H,W,3) | 深度图 (H,W) float32 [0,1] | 1=近景, 0=远景 |
| `compute_depth_mask(image, layers, method)` | BGR (H,W,3) | 蒙版 (H,W) uint8 | 像素值 0~(layers-1) 对应不同深度层 |
| `apply_bokeh(image, aperture, focus_distance, ...)` | BGR (H,W,3) | BGR (H,W,3) | **最常用** — 真实景深虚化 |
| `apply_bokeh_by_layers(image, aperture, layers, focus_layer)` | BGR (H,W,3) | BGR (H,W,3) | 离散分层虚化，省资源 |

#### apply_bokeh 参数详解

```python
CameraSimDepthBridge.apply_bokeh(
    image,              # BGR uint8 (H, W, 3)
    aperture,           # 光圈 f 值: 1.4~22
                        #   f/1.4 → 浅景深（强虚化）
                        #   f/5.6 → 适中
                        #   f/16  → 深景深（几乎无虚化）
    focus_distance=0.5, # 对焦距离: 0~1
                        #   0   = 对焦在最近处
                        #   0.5 = 对焦在中间（默认）
                        #   1   = 对焦在最远处
    depth_smooth_sigma=5.0,  # 深度图平滑度，越大过渡越自然
    max_blur_sigma=None,     # 最大模糊强度
                              # 不传则自动根据光圈计算：
                              #   f/1.4 → ~12, f/2.8 → ~6
                              #   f/5.6 → ~3,  f/11  → ~1.5
)
```

### 底层函数

| 函数 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `depth_mask(input_path, output_path, layers, method, device)` | 文件路径 | 文件路径 | 全流程：文件读入 → 蒙版文件写出 |
| `depth_mask_from_array(image, layers, method, device)` | np.ndarray (H,W,3) | np.ndarray (H,W) | 纯内存运算，不碰磁盘 |
| `get_depth_map(input_path / np.ndarray, device)` | 路径或数组 | np.ndarray (H,W) | 仅获取深度图，不做分割 |

---

## 三、接口协议（Protocol）

CameraSim 的 `interfaces.py` 中定义的 `DepthEngineInterface`：

```python
@runtime_checkable
class DepthEngineInterface(Protocol):
    """景深引擎接口"""

    @staticmethod
    def compute_depth_map(image: NDArray) -> NDArray:
        """返回 (H, W) float32, [0, 1], 1=近景"""
        ...

    def compute_depth_mask(self, image: NDArray, layers: int = 3) -> NDArray:
        """返回 (H, W) uint8 多层蒙版"""
        ...

    def apply_bokeh(self, image: NDArray, aperture: float,
                    focus_distance: float = 0.5) -> NDArray:
        """返回 BGR uint8 虚化结果"""
        ...
```

`CameraSimDepthBridge` 实现了此 Protocol，可作为依赖注入：

```python
from interfaces import DepthEngineInterface
from depth_mask_engine import CameraSimDepthBridge

engine: DepthEngineInterface = CameraSimDepthBridge()
result = engine.apply_bokeh(img, aperture=2.8)
```

---

## 四、架构说明

### 数据流

```
CameraSim ImageProcessor
        │
        ▼
CameraSimDepthBridge.apply_bokeh()    ← 替换原 simulate_aperture_blur()
        │
        ├── compute_depth_map()       深度学习模型估算实际深度
        ├── 计算离焦量 (depth - focus_distance)
        └── 多尺度分层高斯模糊
                │
                ▼
        返回真实景深虚化图像
```

### 与原有 radial blur 的区别

| 特性 | 原 simulate_aperture_blur | apply_bokeh (本接口) |
|------|--------------------------|---------------------|
| 模糊依据 | 距画面中心距离 | **每像素实际深度** |
| 等距物体 | 同一模糊度 | 根据距离焦点远近不同 |
| 前/背景 | 不区分 | **前景清晰/背景模糊**真实还原 |
| 性能 | 实时 (~5ms) | 稍慢 (~200-500ms) 含模型推理 |

---

## 五、常见用法

### 在 CameraSim UI 中切换景深模式

```python
# camera_ui.py
from depth_mask_engine import CameraSimDepthBridge
from image_processor import process as old_process

def process_with_bokeh(img, aperture, ev_diff, iso):
    img = CameraSimDepthBridge.apply_bokeh(img, aperture)
    img = old_process(img, aperture, ev_diff, iso)
    return img
```

### 单独获取深度图用于调试

```python
depth = CameraSimDepthBridge.compute_depth_map(image_bgr)
# depth: (H, W), float32, [0, 1]
# 可视化：
cv2.imshow("depth", depth)  # 近处亮，远处暗
```

### 获取离散分层用于逐层处理

```python
mask = CameraSimDepthBridge.compute_depth_mask(image_bgr, layers=5)
# 分别处理每一层
for layer_id in range(5):
    layer_mask = (mask == layer_id).astype(np.uint8) * 255
    cv2.imshow(f"layer_{layer_id}", layer_mask)
```

---

## 六、性能说明

- **首次调用**：加载深度学习模型，约 1-3 秒
- **后续调用**：模型已缓存，仅推理时间，约 100-300ms（取决于图像分辨率）
- **apply_bokeh_by_layers()** 比 apply_bokeh() 快，适合实时预览
- 建议对用户上传图片做 **max_dim=800** 缩放后调用，大幅提升速度
