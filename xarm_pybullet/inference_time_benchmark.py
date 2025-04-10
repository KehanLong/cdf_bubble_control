import torch
import numpy as np
import time
from robot_cdf import RobotCDF
from self_collision_cdf import SelfCollisionCDF

class CombinedCDFBenchmark:
    def __init__(self, device='cuda'):
        self.device = device
        self.robot_cdf = RobotCDF(device)
        self.self_collision_cdf = SelfCollisionCDF(device)
    
    def query_combined_cdf(self, points, joint_angles, return_gradients=False):
        """
        Query both CDFs and return their minimum
        Args:
            points: [B, N, 3] tensor of points
            joint_angles: [B, 6] tensor of joint angles
            return_gradients: bool, whether to return gradients
        Returns:
            min_cdf_values: [B, N] tensor of minimum CDF values
            gradients: tuple of gradients if return_gradients=True
        """
        # Query workspace CDF
        workspace_cdf =  self.robot_cdf.query_cdf(
            points, joint_angles, return_gradients=return_gradients
        )
        
        # Query self-collision CDF
        self_collision_cdf = self.self_collision_cdf.query_cdf(
            joint_angles, return_gradients=return_gradients
        )
        
        # Expand self-collision CDF to match workspace CDF shape
        self_collision_cdf = self_collision_cdf.unsqueeze(1).expand(-1, points.shape[1])
        
        # Take minimum of both CDFs
        min_cdf_values = torch.minimum(workspace_cdf, self_collision_cdf)
        
        if return_gradients:
            return min_cdf_values, (workspace_grad, self_collision_grad)
        return min_cdf_values

    def benchmark_inference_time(self, q_sizes, p_sizes, num_trials=100):
        """
        Benchmark combined inference time for different batch sizes
        Args:
            q_sizes: List of configuration batch sizes
            p_sizes: List of point cloud sizes
            num_trials: Number of trials to average over
        """
        results = {}
        
        for q_size in q_sizes:
            for p_size in p_sizes:
                # Generate random test data
                joint_angles = torch.rand(q_size, 6, device=self.device) * 2 * np.pi - np.pi
                points = torch.rand(q_size, p_size, 3, device=self.device) * 2 - 1
                
                # Warm-up runs
                for _ in range(10):
                    _ = self.query_combined_cdf(points, joint_angles)
                
                # Timing runs
                torch.cuda.synchronize()
                total_times = []
                workspace_times = []
                self_collision_times = []
                
                for _ in range(num_trials):
                    # Time workspace CDF
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    _ = self.robot_cdf.query_cdf(points, joint_angles)
                    torch.cuda.synchronize()
                    workspace_times.append(time.perf_counter() - start)
                    
                    # Time self-collision CDF
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    _ = self.self_collision_cdf.query_cdf(joint_angles)
                    torch.cuda.synchronize()
                    self_collision_times.append(time.perf_counter() - start)
                    
                    # Time combined query
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    _ = self.query_combined_cdf(points, joint_angles)
                    torch.cuda.synchronize()
                    total_times.append(time.perf_counter() - start)
                
                # Convert to milliseconds and compute statistics
                total_times = np.array(total_times) * 1000
                workspace_times = np.array(workspace_times) * 1000
                self_collision_times = np.array(self_collision_times) * 1000
                
                results[(q_size, p_size)] = {
                    'total': {
                        'mean': np.mean(total_times),
                        'std': np.std(total_times)
                    },
                    'workspace': {
                        'mean': np.mean(workspace_times),
                        'std': np.std(workspace_times)
                    },
                    'self_collision': {
                        'mean': np.mean(self_collision_times),
                        'std': np.std(self_collision_times)
                    }
                }
        
        return results

def print_benchmark_results(results):
    """Pretty print the benchmark results"""
    print("\nInference Time Results (milliseconds):")
    print("=" * 80)
    print(f"{'Batch Sizes':^20} | {'Total':^18} | {'Workspace':^18} | {'Self-Collision':^18}")
    print("-" * 80)
    
    for (q_size, p_size), stats in sorted(results.items()):
        print(
            f"q={q_size:3d}, p={p_size:5d} | "
            f"{stats['total']['mean']:6.3f} ± {stats['total']['std']:5.3f} | "
            f"{stats['workspace']['mean']:6.3f} ± {stats['workspace']['std']:5.3f} | "
            f"{stats['self_collision']['mean']:6.3f} ± {stats['self_collision']['std']:5.3f}"
        )

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    benchmark = CombinedCDFBenchmark(device)
    
    # Define test sizes
    q_sizes = [1, 10, 100]  # Configuration batch sizes
    p_sizes = [1, 10, 100, 1000, 10000]  # Point cloud sizes
    
    print(f"Running benchmark on device: {device}")
    print(f"Configuration batch sizes: {q_sizes}")
    print(f"Point cloud sizes: {p_sizes}")
    
    # Run benchmark
    results = benchmark.benchmark_inference_time(q_sizes, p_sizes)
    
    # Print results
    print_benchmark_results(results) 