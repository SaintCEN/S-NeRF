import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# 生成合成的卫星图像数据（模拟）
def generate_synthetic_data(num_images, image_size, true_shape):
    images = []
    depths = []
    for _ in range(num_images):
        # 模拟不同的光照条件（这里简化为随机强度）
        light_intensity = np.random.uniform(0.5, 1.5)
        # 模拟图像（这里简化为与真实形状和光照相关的随机噪声）
        image = true_shape + np.random.normal(0, 0.1, image_size) * light_intensity
        images.append(image)
        # 模拟深度图（这里简化为与真实形状相关的随机噪声）
        depth = true_shape + np.random.normal(0, 0.5, image_size)
        depths.append(depth)
    return np.array(images), np.array(depths)


# 定义真实形状（模拟地球表面的简单地形）
true_shape = np.zeros((100, 100))
true_shape[30:70, 30:70] = 1.0

# 生成合成数据
num_images = 20
image_size = (100, 100)
images, depths = generate_synthetic_data(num_images, image_size, true_shape)
images = torch.from_numpy(images).float().unsqueeze(1)
depths = torch.from_numpy(depths).float().unsqueeze(1)
torch.Size([20, 1, 100, 100])


def get_snerf_input(image):
    h, w = image.shape[-2:]
    x_coords, y_coords = torch.meshgrid(torch.arange(w), torch.arange(h), indexing='ij')
    coords = torch.stack([x_coords, y_coords, torch.zeros_like(x_coords)], dim=-1).float()
    return coords.view(-1, 3)

x = get_snerf_input(images[0])
print(x)

# 定义S-NeRF模型
class SNeRF(nn.Module):
    def __init__(self):
        super(SNeRF, self).__init__()
        # 定义密度网络
        self.density_layers = nn.Sequential(
            nn.Linear(3, 64),  # 输入 3D 坐标
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        # 定义反照率网络
        self.albedo_layers = nn.Sequential(
            nn.Linear(3, 64),  # 输入 3D 坐标
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        # 定义太阳可见性网络
        self.sun_visibility_layers = nn.Sequential(
            nn.Linear(3 + 2, 64),  # 输入 3D 坐标 + 太阳方向（合成后的5维输入）
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        # 定义天空颜色估计层
        self.sky_color_layers = nn.Sequential(
            nn.Linear(2, 64),  # 输入太阳方向（2维）
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x, sun_direction):
        # 修正：拼接 x 和 sun_direction 时，确保维度正确
        density_input = x  # 只使用 3D 坐标输入到密度层
        density = self.density_layers(density_input)

        albedo_input = x  # 只使用 3D 坐标输入到反照率层
        albedo = self.albedo_layers(albedo_input)

        # 太阳可见性和天空颜色的输入：拼接 3D 坐标和 太阳方向
        # 计算天空颜色：仅使用太阳方向

        sky_color = self.sky_color_layers(sun_direction)
        sun_direction = sun_direction.unsqueeze(0).expand(x.shape[0], -1)  # 扩展为与 x 的第0维相同

        visibility_input = torch.cat([x, sun_direction], dim=-1)  # 拼接

        sun_visibility = self.sun_visibility_layers(visibility_input)

        return density, albedo, sun_visibility, sky_color


model = snerf_model = SNeRF()
density, albedo, sun_visibility, sky_color = model(x, torch.tensor([np.pi / 4, np.pi / 4]))
print(density)
print(albedo)
print(sun_visibility)
print(sky_color)

irradiance = sun_visibility * torch.ones(1) + (1 - sun_visibility) * sky_color
predicted_color = albedo * irradiance
print(images[0][0].shape)
print(predicted_color.shape)
loss = nn.MSELoss()(predicted_color, images[0][0].view(10000, -1))
print(loss)
torch.Size([100, 100])
torch.Size([10000, 1])

print(sun_visibility.shape)
print(images[0].shape)
torch.Size([10000, 1])
torch.Size([1, 100, 100])

model = snerf_model = SNeRF()
sun_directions = [torch.tensor([np.random.uniform(0, 2 * np.pi), np.random.uniform(0, np.pi)]).float() for _ in
                  range(len(images))]

# 定义S-NeRF损失函数
def snerf_loss(snerf_model, images, sun_directions, lambda_s=0.0005):
    total_loss = 0.0
    batch_size = images.shape[0]

    # 计算像素损失
    for i in range(batch_size):
        x = get_snerf_input(images[i])  # 获取S-NeRF模型的输入
        density, albedo, sun_visibility, sky_color = snerf_model(x, sun_directions[i])
        irradiance = sun_visibility * torch.ones(3) + (1 - sun_visibility) * sky_color
        predicted_color = albedo[i] * irradiance
        total_loss += nn.MSELoss()(predicted_color, images[i][0].view(10000, -1))
    # 计算太阳校正损失
    for i in range(batch_size):
        x = get_snerf_input(images[i])
        density, albedo, sun_visibility, sky_color = snerf_model(x, sun_directions[i])
        with torch.no_grad():
            transparency = compute_transparency(density)  # 假设已经定义了计算透明度的函数
        solar_correction_loss = lambda_s * (torch.sum(transparency - sun_visibility[i]) ** 2) + 1 - torch.sum(
            sun_visibility[i] * transparency)
        total_loss += solar_correction_loss
    return total_loss / batch_size


# 计算透明度的函数（根据论文中的公式）
def compute_transparency(density):
    alpha = 1 - torch.exp(-density)
    alpha = torch.clamp(alpha, min=1e-6, max=0.99)  # 限制 alpha 的范围
    transparency = torch.cumprod(1 - alpha, dim=0)
    return transparency


print(snerf_loss(snerf_model, images, sun_directions))

# 初始化S-NeRF模型
snerf_model = SNeRF()

# 训练S-NeRF模型
num_epochs = 100
learning_rate = 0.01
optimizer = optim.Adam(snerf_model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    optimizer.zero_grad()
    sun_directions = [torch.tensor([np.random.uniform(0, 2 * np.pi), np.random.uniform(0, np.pi)]).float() for _ in
                      range(len(images))]
    loss = snerf_loss(snerf_model, images, sun_directions)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f'Epoch {epoch + 1}/{num_epochs}, S-NeRF Loss: {loss.item()}')