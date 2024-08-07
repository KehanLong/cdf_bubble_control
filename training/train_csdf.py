# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader

'''
training without eikonal constrainy
'''

# def train(net, dataloader, val_dataloader, num_epochs, learning_rate, device, loss_threshold=1e-4, lambda_eikonal=0.1):
#     criterion = nn.MSELoss()
#     optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

#     net.to(device)

#     for epoch in range(num_epochs):
#         net.train()
#         running_loss = 0.0

#         for inputs, targets in dataloader:
#             inputs, targets = inputs.to(device), targets.to(device)

#             optimizer.zero_grad()
#             outputs = net(inputs)
#             loss = criterion(outputs, targets)
#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item()

#         epoch_loss = running_loss / len(dataloader)
#         print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss:.4f}")

#         # Evaluate the model on the validation set
#         net.eval()
#         with torch.no_grad():
#             val_loss = 0.0
#             for val_inputs, val_targets in val_dataloader:
#                 val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
#                 val_outputs = net(val_inputs)
#                 val_loss += criterion(val_outputs, val_targets).item()

#             val_loss /= len(val_dataloader)
#             print(f"Validation Loss: {val_loss:.4f}")

#         # Check if both training loss and validation loss are smaller than the threshold
#         if epoch_loss < loss_threshold and val_loss < loss_threshold:
#             print(f"Training stopped early at epoch {epoch+1} as both losses are below the threshold.")
#             break

#     return net


import jax
import jax.numpy as jnp
from flax.training import train_state
import optax
from flax import linen as nn

'''
if no GPU
'''
jax.config.update('jax_platform_name', 'cpu')



def train(net, train_dataset, val_dataset, num_epochs, learning_rate, loss_threshold=1e-4, lambda_eikonal=0.1):
    def loss_fn(params, batch):
        inputs, targets = batch
        outputs = net.apply(params, inputs)
        loss = jnp.mean((outputs - targets) ** 2)
        return loss
    
    @jax.jit
    def train_step(state, batch):
        loss, grads = jax.value_and_grad(loss_fn)(state.params, batch)
        state = state.apply_gradients(grads=grads)
        return state, loss
    
    tx = optax.adam(learning_rate=learning_rate)
    state = train_state.TrainState.create(
        apply_fn=net.apply, params=net.init(jax.random.PRNGKey(0), jnp.zeros((1, 4 * 3 + 3))), tx=tx
    )
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for i in range(len(train_dataset)):
            inputs, targets = train_dataset[i]
            state, loss = train_step(state, (inputs, targets))
            epoch_loss += loss
            
        epoch_loss /= len(train_dataset)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss:.4f}")
        
        # Evaluate the model on the validation set
        val_loss = 0.0
        for i in range(len(val_dataset)):
            val_inputs, val_targets = val_dataset[i]
            val_outputs = net.apply(state.params, val_inputs)
            val_loss += jnp.mean((val_outputs - val_targets) ** 2)
        val_loss /= len(val_dataset)
        print(f"Validation Loss: {val_loss:.4f}")
        
        # Check if both training loss and validation loss are smaller than the threshold
        if epoch_loss < loss_threshold and val_loss < loss_threshold:
            print(f"Training stopped early at epoch {epoch+1} as both losses are below the threshold.")
            break
    
    return state.params