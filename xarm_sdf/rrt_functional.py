import numpy as np
import torch
from typing import List, Tuple, Optional
from scipy.spatial import cKDTree
import time
import logging

logger = logging.getLogger(__name__)

class RRTBasePlanner:
    def __init__(self, 
                 robot_sdf,
                 robot_fk,
                 joint_limits: Tuple[np.ndarray, np.ndarray],
                 step_size: float = 0.1,
                 max_iter: int = 10000,
                 batch_size: int = 1000,
                 goal_bias: float = 0.1,
                 safety_margin: float = 0.02,
                 device: str = 'cuda'):
        self.robot_sdf = robot_sdf
        self.robot_fk = robot_fk
        self.joint_lower_limits = joint_limits[0]
        self.joint_upper_limits = joint_limits[1]
        self.step_size = step_size
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.goal_bias = goal_bias
        self.safety_margin = safety_margin
        self.device = device

    def _get_uniform_random_configs(self, n_samples: int, rng=None) -> np.ndarray:
        """Generate batch of random configurations"""
        if rng is None:
            rng = np.random.default_rng()
        return rng.uniform(self.joint_lower_limits, self.joint_upper_limits, size=(n_samples, len(self.joint_lower_limits)))

    def _check_collision_batch(self, configs: np.ndarray, obstacle_points: Optional[torch.Tensor] = None) -> np.ndarray:
        """Check collision for batch of configurations using RobotSDF"""
        configs_tensor = torch.tensor(configs, device=self.device)
        
        # Get SDF values for all configurations
        if obstacle_points is not None:
            # Ensure obstacle_points is the right shape [N, 3]
            if obstacle_points.dim() == 4:
                obstacle_points = obstacle_points.squeeze(0)  # Remove extra dimensions
            
            # Query SDF values for each configuration
            sdf_values = self.robot_sdf.query_sdf(
                points=obstacle_points.expand(len(configs), -1, -1),  # [B, N, 3]
                joint_angles=configs_tensor,  # [B, 6]
                return_gradients=False
            )
            # Configuration is valid if minimum SDF value is above safety margin
            min_sdf_values = sdf_values.min(dim=1)[0]  # Get minimum across all points
            return min_sdf_values > self.safety_margin
        return torch.ones(len(configs), dtype=torch.bool, device=self.device)

    def _extend_towards(self, from_config: np.ndarray, to_config: np.ndarray) -> np.ndarray:
        """Extend from one configuration towards another by step_size"""
        diff = to_config - from_config
        dist = np.linalg.norm(diff)
        if dist > self.step_size:
            diff = diff / dist * self.step_size
        return from_config + diff

    def _get_ee_position(self, config: np.ndarray) -> np.ndarray:
        """Get end-effector position for configuration"""
        config_tensor = torch.tensor(config, device=self.device, dtype=torch.float32)
        if len(config_tensor.shape) == 1:
            config_tensor = config_tensor.unsqueeze(0)
        ee_pos = self.robot_fk.fkine(config_tensor)
        return ee_pos.squeeze().cpu().numpy()

class RRTPlanner(RRTBasePlanner):
    def plan(self, 
            start_config: np.ndarray,
            goal_config: np.ndarray,
            obstacle_points: Optional[torch.Tensor] = None,
            rng=None) -> List[np.ndarray]:
        """Plan path using standard RRT"""
        if rng is None:
            rng = np.random.default_rng()

        # Convert inputs to float32 if needed
        start_config = start_config.astype(np.float32)
        goal_config = goal_config.astype(np.float32)

        start_time = time.time()
        print("\nStarting RRT planning...")
        
        # Initialize tree with start configuration
        tree = [(start_config, -1)]
        tree_kdtree = cKDTree(np.array([start_config]))
        
        # Statistics tracking
        closest_dist_to_goal = float('inf')
        n_nodes = 1
        
        for iter_idx in range(self.max_iter):
            # Generate batch of samples
            goal_biased_count = int(self.batch_size * self.goal_bias)
            regular_count = self.batch_size - goal_biased_count
            
            # Generate random samples and add goal bias
            random_configs = self._get_uniform_random_configs(regular_count, rng)
            
            # Add goal-biased samples
            if goal_biased_count > 0:
                goal_biased_configs = np.tile(goal_config, (goal_biased_count, 1))
                random_configs = np.vstack([random_configs, goal_biased_configs])
            
            # Process batch
            distances, nearest_indices = tree_kdtree.query(random_configs, k=1)
            new_configs = np.array([
                self._extend_towards(tree[idx][0], sample)
                for sample, idx in zip(random_configs, nearest_indices)
            ])
            
            # Check collisions for all new configurations
            valid_mask = self._check_collision_batch(new_configs, obstacle_points)
            
            # Add valid configurations to tree
            for config, nearest_idx, is_valid in zip(new_configs, nearest_indices, valid_mask):
                if is_valid:
                    tree.append((config, nearest_idx))
                    n_nodes += 1
                    
                    # Check if reached goal
                    dist_to_goal = np.linalg.norm(config - goal_config)
                    closest_dist_to_goal = min(closest_dist_to_goal, dist_to_goal)
                    
                    if dist_to_goal < 0.1:  # threshold in configuration space
                        path = self._extract_path(tree, len(tree)-1)
                        elapsed_time = time.time() - start_time
                        print(f"\nPath found!")
                        print(f"Time: {elapsed_time:.2f}s")
                        print(f"Iterations: {iter_idx + 1}")
                        print(f"Tree size: {n_nodes} nodes")
                        print(f"Path length: {len(path)} waypoints")
                        return path
            
            # Update KD-tree
            tree_kdtree = cKDTree(np.array([node[0] for node in tree]))
            
            # Log progress more frequently
            if iter_idx % 2 == 0:
                print(f"Iter: {iter_idx}, Nodes: {n_nodes}, Best dist: {closest_dist_to_goal:.4f}")
        
        elapsed_time = time.time() - start_time
        print(f"\nPlanning failed after {elapsed_time:.2f}s")
        print(f"Tree size: {n_nodes} nodes")
        print(f"Best distance to goal: {closest_dist_to_goal:.4f}")
        
        return []

    def _extract_path(self, tree: List[Tuple], goal_idx: int) -> List[np.ndarray]:
        """Extract path from tree from start to goal"""
        path = []
        current_idx = goal_idx
        while current_idx != -1:
            path.append(tree[current_idx][0])
            current_idx = tree[current_idx][1]
        return list(reversed(path)) 
    

class RRTConnectPlanner(RRTBasePlanner):
    def plan(self,
            start_config: np.ndarray,
            goal_config: np.ndarray,
            obstacle_points: Optional[torch.Tensor] = None,
            rng=None) -> List[np.ndarray]:
        """
        Plan path using RRT-Connect
        """
        if rng is None:
            rng = np.random.default_rng()

        start_time = time.time()
        
        # Initialize both trees
        start_tree = [(start_config, -1)]
        goal_tree = [(goal_config, -1)]
        
        start_kdtree = cKDTree(np.array([start_config]))
        goal_kdtree = cKDTree(np.array([goal_config]))
        
        for iter_idx in range(self.max_iter):
            # Alternate between trees
            if iter_idx % 2 == 0:
                success, path = self._extend_tree(
                    start_tree, start_kdtree,
                    goal_tree, goal_kdtree,
                    obstacle_points, rng
                )
            else:
                success, path = self._extend_tree(
                    goal_tree, goal_kdtree,
                    start_tree, start_kdtree,
                    obstacle_points, rng
                )
                if success:
                    path = list(reversed(path))
            
            if success:
                logger.info(f"Trees connected after {iter_idx} iterations and {time.time() - start_time:.2f}s")
                return path
            
            if iter_idx % 10 == 0:
                logger.info(f"Iteration {iter_idx}, Trees size: {len(start_tree)}, {len(goal_tree)}")
        
        logger.warning("Failed to find path")
        return []

    def _extend_tree(self, 
                    growing_tree: List[Tuple],
                    growing_kdtree: cKDTree,
                    target_tree: List[Tuple],
                    target_kdtree: cKDTree,
                    obstacle_points: Optional[torch.Tensor],
                    rng) -> Tuple[bool, List[np.ndarray]]:
        """Extend growing_tree and try to connect to target_tree"""
        # Generate batch of samples
        random_configs = self._get_uniform_random_configs(self.batch_size, rng)
        
        # Find nearest neighbors
        distances, nearest_indices = growing_kdtree.query(random_configs, k=1)
        
        # Extend towards samples
        new_configs = np.array([
            self._extend_towards(growing_tree[idx][0], sample)
            for sample, idx in zip(random_configs, nearest_indices)
        ])
        
        # Check collisions
        valid_mask = self._check_collision_batch(new_configs, obstacle_points)
        
        # Try to connect trees
        for config, nearest_idx, is_valid in zip(new_configs, nearest_indices, valid_mask):
            if is_valid:
                growing_tree.append((config, nearest_idx))
                
                # Find nearest config in target tree
                target_dist, target_idx = target_kdtree.query(config.reshape(1, -1), k=1)
                
                if target_dist < self.step_size:
                    # Trees connected! Extract path
                    return True, self._extract_connecting_path(
                        growing_tree, len(growing_tree)-1,
                        target_tree, target_idx[0]
                    )
        
        return False, []

    def _extract_connecting_path(self,
                               tree1: List[Tuple],
                               idx1: int,
                               tree2: List[Tuple],
                               idx2: int) -> List[np.ndarray]:
        """Extract path connecting two trees"""
        path1 = []
        current_idx = idx1
        while current_idx != -1:
            path1.append(tree1[current_idx][0])
            current_idx = tree1[current_idx][1]
        path1 = list(reversed(path1))
        
        path2 = []
        current_idx = idx2
        while current_idx != -1:
            path2.append(tree2[current_idx][0])
            current_idx = tree2[current_idx][1]
        
        return path1 + path2