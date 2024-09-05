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

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model_dict = {}
    

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_eikonal = 0.0
        epoch_tension = 0.0

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs.requires_grad = True  # Ensure inputs require gradients

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs).squeeze(-1)

            # Compute MSE loss
            mse_loss = torch.nn.functional.mse_loss(outputs, targets)

            # Compute Eikonal loss for configuration gradients
            grad_outputs = torch.ones_like(outputs)
            gradients = torch.autograd.grad(outputs, inputs, grad_outputs=grad_outputs, create_graph=True)[0]
            
            # Extract gradients for the original configuration parameters (NUM_LINKS )
            config_gradients = gradients[:, :NUM_LINKS]
            eikonal_loss = torch.mean((config_gradients.norm(2, dim=-1) - 1)**2)

            # Compute Tension loss
            dd_gradients = torch.autograd.grad(gradients, inputs, grad_outputs=torch.ones_like(gradients), create_graph=True)[0]
            tension_loss = dd_gradients.square().sum(dim=-1).mean()

            # Combine losses
            w0, w1, w2 = 1.0, 0.02, 0.01  # Adjust these weights as needed
            loss = w0 * mse_loss + w1 * eikonal_loss  # Only including MSE and Eikonal for now


            # Backward pass and optimize
            loss.backward()
            

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_mse += mse_loss.item()
            epoch_eikonal += eikonal_loss.item()
            epoch_tension += tension_loss.item()

        epoch_loss /= len(dataloader)
        epoch_mse /= len(dataloader)
        epoch_eikonal /= len(dataloader)
        epoch_tension /= len(dataloader)

        if epoch == 1 or (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}: Loss: {epoch_loss:.4f}, MSE: {epoch_mse:.4f}, Eikonal: {epoch_eikonal:.4f}, Tension: {epoch_tension:.4f}")

        scheduler.step(epoch_loss)

        if (epoch + 1) % 100 == 0:
            model_dict[epoch] = model.state_dict()
            torch.save(model_dict, os.path.join('trained_models/cdf_models', 'cdf_model_dict.pt'))

    return model
