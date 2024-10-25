import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import grad

class ImplicitNet(nn.Module):
    def __init__(
        self,
        d_in,
        dims,
        skip_in=(),
        geometric_init=True,
        radius_init=1,
        beta=100,
        use_skip_connections=True
    ):
        super().__init__()

        dims = [d_in] + dims + [1]

        self.num_layers = len(dims)
        self.skip_in = skip_in if use_skip_connections else ()

        for layer in range(0, self.num_layers - 1):

            if layer + 1 in self.skip_in:
                out_dim = dims[layer + 1] - d_in
            else:
                out_dim = dims[layer + 1]

            lin = nn.Linear(dims[layer], out_dim)

            if geometric_init:
                if layer == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[layer]), std=0.00001)
                    torch.nn.init.constant_(lin.bias, -radius_init)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
            else:
                torch.nn.init.kaiming_normal_(lin.weight, mode='fan_in', nonlinearity='relu')
                torch.nn.init.constant_(lin.bias, 0)

            setattr(self, "lin" + str(layer), lin)

        self.activation = nn.Softplus(beta=beta) if beta > 0 else nn.ReLU()

    def forward(self, input):
        x = input
        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))
            if layer in self.skip_in:
                x = torch.cat([x, input], -1) / np.sqrt(2)
            x = lin(x)
            if layer < self.num_layers - 2:
                x = self.activation(x)
        return x

class ImprovedSampler:
    def __init__(self, global_sigma, local_sigma=0.01):
        self.global_sigma = global_sigma
        self.local_sigma = local_sigma

    def get_points(self, points):
        batch_size, num_points, dim = points.shape
        sample_local = points + torch.randn_like(points) * self.local_sigma
        sample_global = torch.rand(batch_size, num_points // 2, dim, device=points.device) * (self.global_sigma * 2) - self.global_sigma
        sample = torch.cat([sample_local, sample_global], dim=1)
        return sample

def generate_ellipse_dataset(num_points, a=1.0, b=0.5, center_x=1.0, center_y=1.0):
    theta = np.linspace(0, 2*np.pi, num_points, endpoint=False)
    x = center_x + a * np.cos(theta)
    y = center_y + b * np.sin(theta)
    points = np.column_stack((x, y))
    
    normals = np.column_stack((x - center_x, y - center_y))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    
    return points, normals


def train_sdf_net(model, optimizer, num_epochs, batch_size, points, normals, sampler, early_stop_threshold=1e-3):
    for epoch in range(num_epochs):
        indices = np.random.choice(len(points), batch_size, replace=False)
        batch_points = torch.FloatTensor(points[indices]).unsqueeze(0).cuda()
        batch_normals = torch.FloatTensor(normals[indices]).cuda()
        
        nonmnfld_points = sampler.get_points(batch_points).cuda()
        
        batch_points = batch_points.squeeze(0).requires_grad_()
        nonmnfld_points = nonmnfld_points.squeeze(0).requires_grad_()
        
        mnfld_pred = model(batch_points)
        nonmnfld_pred = model(nonmnfld_points)
        
        # Use PyTorch's autograd for gradient computation
        mnfld_grad = torch.autograd.grad(mnfld_pred.sum(), batch_points, create_graph=True)[0]
        nonmnfld_grad = torch.autograd.grad(nonmnfld_pred, nonmnfld_points, grad_outputs=torch.ones_like(nonmnfld_pred), create_graph=True)[0]


        # Compute the new GD2S loss
        gd2s_points = nonmnfld_points - nonmnfld_pred.detach() * nonmnfld_grad
        gd2s_pred = model(gd2s_points)
        gd2s_loss = torch.mean(gd2s_pred.abs())
        
        mnfld_loss = torch.mean(mnfld_pred.abs())
        eikonal_loss = torch.mean((nonmnfld_grad.norm(2, dim=-1) - 1) ** 2)
        normal_loss = torch.mean((mnfld_grad - batch_normals).abs().norm(2, dim=1))
        
        # Add the new GD2S loss to the total loss
        loss = mnfld_loss + 0.1 * eikonal_loss + 0.1 * gd2s_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, "
                  f"Manifold Loss: {mnfld_loss.item():.4f}, Eikonal Loss: {eikonal_loss.item():.4f}, "
                  f"Normal Loss: {normal_loss.item():.4f}, GD2S Loss: {gd2s_loss.item():.4f}")
        
        # Early stopping condition
        if loss.item() < early_stop_threshold:
            print(f"Early stopping at epoch {epoch+1}. Loss: {loss.item():.6f}")
            break
    
    # If we didn't break the loop, it means we completed all epochs
    else:
        print(f"Training completed after {num_epochs} epochs. Final loss: {loss.item():.6f}")

    return epoch + 1  # Return the number of epochs actually trained

def plot_sdf(model, points, normals, a=1.0, b=0.5, center_x=1.0, center_y=1.0, num_normals_to_plot=20):
    model.cpu()  # Ensure the model is on CPU
    x = np.linspace(-1, 3, 200)
    y = np.linspace(-1, 3, 200)
    X, Y = np.meshgrid(x, y)
    
    grid_points = np.column_stack((X.ravel(), Y.ravel()))
    with torch.no_grad():
        Z = model(torch.FloatTensor(grid_points)).numpy().reshape(X.shape)
    
    plt.figure(figsize=(12, 10))
    plt.contourf(X, Y, Z, levels=20, cmap='RdBu')
    plt.colorbar(label='SDF Value')
    plt.title('Learned SDF of an Ellipse with Normals')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    # Plot the zero level set
    plt.contour(X, Y, Z, levels=[0], colors='k', linewidths=2)
    
    # Plot the true ellipse
    theta = np.linspace(0, 2*np.pi, 100)
    x_true = center_x + a * np.cos(theta)
    y_true = center_y + b * np.sin(theta)
    plt.plot(x_true, y_true, 'r--', linewidth=2, label='True Ellipse')
    
    # Plot a subset of the computed normals
    # indices = np.linspace(0, len(points) - 1, num_normals_to_plot, dtype=int)
    # normal_scale = 0.2
    # plt.quiver(points[indices, 0], points[indices, 1], 
    #            normals[indices, 0], normals[indices, 1], 
    #            color='g', scale=1/normal_scale, width=0.003, 
    #            label='Computed Normals')
    
    plt.legend()
    plt.axis('equal')
    plt.savefig('learned_sdf_ellipse_with_normals.png')
    plt.close()

if __name__ == "__main__":
    # Hyperparameters
    num_points = 2000
    num_epochs = 2000
    batch_size = 128
    learning_rate = 0.0015
    a, b = 1.0, 0.5
    center_x, center_y = 1.0, 1.0

    points, normals = generate_ellipse_dataset(num_points, a, b, center_x, center_y)

    model = ImplicitNet(d_in=2, dims=[512, 512, 512, 512], skip_in=[4], geometric_init=True).cuda()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    sampler = ImprovedSampler(global_sigma=2.0, local_sigma=0.01)

    epochs_trained = train_sdf_net(model, optimizer, num_epochs, batch_size, points, normals, sampler)

    model.cpu()  # Move the model to CPU before plotting
    plot_sdf(model, points, normals, a, b, center_x, center_y, num_normals_to_plot=20)
    print("SDF plot with normals saved as 'learned_sdf_ellipse_with_normals.png'")
