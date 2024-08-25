
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax

'''
if no GPU
'''
# jax.config.update('jax_platform_name', 'cpu')

def train(net, dataset, num_epochs, learning_rate, lambda_eikonal=0.1, threshold=1e-3):
    def loss_fn(params, batch):
        points, distances = batch
        pred_distances = net.apply(params, points)
        loss = jnp.mean((pred_distances - distances) ** 2)
        
        # Compute the Eikonal loss
        grad_fn = jax.grad(lambda x: net.apply(params, x).sum())
        gradient = jax.vmap(grad_fn)(points)
        eikonal_loss = jnp.mean((jnp.linalg.norm(gradient, axis=1) - 1) ** 2)
        
        # Combine the losses
        total_loss = loss + lambda_eikonal * eikonal_loss
        return total_loss, eikonal_loss
    
    @jax.jit
    def train_step(state, batch):
        (loss, eikonal_loss), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params, batch)
        state = state.apply_gradients(grads=grads)
        return state, loss, eikonal_loss
    
    tx = optax.adam(learning_rate=learning_rate)
    state = train_state.TrainState.create(
        apply_fn=net.apply, params=net.init(jax.random.PRNGKey(0), jnp.zeros((1, 3))), tx=tx
    )
    
    batch_size = 128
    num_batches = len(dataset) // batch_size
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_eikonal_loss = 0.0
        
        for i in range(num_batches):
            batch = dataset[i * batch_size: (i + 1) * batch_size]
            state, loss, eikonal_loss = train_step(state, batch)
            epoch_loss += loss
            epoch_eikonal_loss += eikonal_loss
        
        epoch_loss /= num_batches
        epoch_eikonal_loss /= num_batches
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Eikonal Loss: {epoch_eikonal_loss:.4f}")
        
        if epoch_loss < threshold:
            print(f"Reached loss threshold of {threshold}. Stopping training.")
            break
    
    return state.params

