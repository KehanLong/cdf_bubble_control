import numpy as np
from pathlib import Path



def refine_contact_db(input_path, output_path, min_configs=100):
    """
    Refine contact database by removing points with too few configurations.
    Each configuration is a complete 6-DOF joint configuration.
    """
    print(f"Loading contact database from: {input_path}")
    db = np.load(input_path, allow_pickle=True).item()
    
    # Print initial stats
    print("\nInitial Statistics:")
    print(f"Total points: {len(db['points'])}")
    print(f"Configurations per point: min={min(len(c) for c in db['contact_configs'])}, "
          f"max={max(len(c) for c in db['contact_configs'])}, "
          f"mean={np.mean([len(c) for c in db['contact_configs']]):.1f}")
    
    # Filter points with too few configurations
    valid_indices = [i for i, configs in enumerate(db['contact_configs']) 
                    if len(configs) >= min_configs]
    
    filtered_db = {
        'points': db['points'][valid_indices],
        'contact_configs': [db['contact_configs'][i] for i in valid_indices],
        'link_indices': [db['link_indices'][i] for i in valid_indices]
    }
    
    print(f"\nAfter filtering (min {min_configs} configs):")
    print(f"Remaining points: {len(filtered_db['points'])}")
    print(f"Configurations per point: min={min(len(c) for c in filtered_db['contact_configs'])}, "
          f"max={max(len(c) for c in filtered_db['contact_configs'])}, "
          f"mean={np.mean([len(c) for c in filtered_db['contact_configs']]):.1f}")
    

    print(f"\nSaving filtered database to: {output_path}")
    np.save(output_path, filtered_db)
    return filtered_db 


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).parent.parent
    input_path = PROJECT_ROOT / "data/cdf_data/bfgs_contact_db.npy"
    output_path = PROJECT_ROOT / "data/cdf_data/refined_bfgs_100_contact_db.npy"
    
    refine_contact_db(
        input_path=input_path,
        output_path=output_path,
        min_configs=100
    )