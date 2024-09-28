import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from data.arm_2d_config import NUM_LINKS

def train(model, dataset, num_epochs=500, batch_size=1024, learning_rate=0.001, device='cuda'):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10,
                                                     threshold=0.01, threshold_mode='rel',
                                                     cooldown=0, min_lr=0, eps=1e-08, verbose=False)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model_dict = {}

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_eikonal = 0.0

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs.requires_grad = True  # Ensure inputs require gradients

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs).squeeze(-1)

            # Compute MSE loss
            mse_loss = torch.nn.functional.mse_loss(outputs, targets)

            # Compute Eikonal loss for joint angle gradients
            grad_outputs = torch.ones_like(outputs)
            gradients = torch.autograd.grad(outputs, inputs, grad_outputs=grad_outputs, create_graph=True)[0]
            
            # Extract gradients for sin(theta) and cos(theta)
            angle_gradients = gradients[:, :2*NUM_LINKS]

            # Reconstruct gradients w.r.t. theta
            theta_gradients = torch.zeros(angle_gradients.shape[0], NUM_LINKS, device=device)
            for i in range(NUM_LINKS):
                sin_grad = angle_gradients[:, i]
                cos_grad = angle_gradients[:, NUM_LINKS + i]
                sin_theta = inputs[:, i]
                cos_theta = inputs[:, NUM_LINKS + i]
                theta_gradients[:, i] = cos_theta * sin_grad - sin_theta * cos_grad

            # Compute eikonal loss
            eikonal_loss = torch.mean((theta_gradients.norm(2, dim=-1) - 1)**2)

            # Combine losses
            w0, w1 = 1.0, 0.02  # Adjust these weights as needed
            loss = w0 * mse_loss + w1 * eikonal_loss

            # Backward pass and optimize
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_mse += mse_loss.item()
            epoch_eikonal += eikonal_loss.item()

        epoch_loss /= len(dataloader)
        epoch_mse /= len(dataloader)
        epoch_eikonal /= len(dataloader)

        if epoch == 1 or (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}: Loss: {epoch_loss:.4f}, MSE: {epoch_mse:.4f}, Eikonal: {epoch_eikonal:.4f}")

        scheduler.step(epoch_loss)

        if (epoch + 1) % 100 == 0:
            model_dict[epoch] = model.state_dict()
            torch.save(model_dict, os.path.join('trained_models/cdf_models', 'cdf_model_dict.pt'))

    return model
