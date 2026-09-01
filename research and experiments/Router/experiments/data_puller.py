import os
import pandas as pd
from datasets import load_dataset, get_dataset_config_names

# Define base output directory
BASE_DIR = "data"

# Updated to use the canonical 'mbpp' repository
DATASETS_TO_PULL = {
    "gsm8k": {"path": "openai/gsm8k", "name": "main", "split": "test"},
    "math_500": {"path": "HuggingFaceH4/MATH-500", "name": "default", "split": "test"},
    "mbpp": {"path": "mbpp", "name": "full", "split": "test"},
    "humanevalplus": {"path": "evalplus/humanevalplus", "name": "default", "split": "test"},
    "ifeval": {"path": "google/IFEval", "name": "default", "split": "train"},
    "mmlu_pro": {"path": "TIGER-Lab/MMLU-Pro", "name": "default", "split": "test"},
}

def pull_standard_datasets():
    """Loops through standard datasets and saves them in their respective folders."""
    for local_name, config in DATASETS_TO_PULL.items():
        # Skip datasets that have already been successfully downloaded
        dataset_dir = os.path.join(BASE_DIR, local_name)
        if os.path.exists(dataset_dir) and len(os.listdir(dataset_dir)) > 0:
            print(f"⏭️ Skipping {local_name}: Already downloaded.")
            continue
            
        print(f"\nDownloading {local_name} from {config['path']}...")
        
        try:
            os.makedirs(dataset_dir, exist_ok=True)
            
            # Load dataset from Hugging Face
            kwargs = {"path": config['path'], "split": config['split']}
            if config['name'] and config['name'] != "default":
                kwargs["name"] = config['name']
                
            dataset = load_dataset(**kwargs)
            df = dataset.to_pandas()
            
            # Define output paths
            parquet_path = os.path.join(dataset_dir, f"{local_name}.parquet")
            jsonl_path = os.path.join(dataset_dir, f"{local_name}.jsonl")
            
            # Save to disk
            df.to_parquet(parquet_path, index=False)
            df.to_json(jsonl_path, orient="records", lines=True)
            
            print(f"✅ Saved {local_name} ({len(df)} rows) to {dataset_dir}/")
            
        except Exception as e:
            print(f"❌ Error downloading {local_name}: {e}")

def pull_bbh_dataset():
    """Big-Bench Hard (BBH) fix: The splits are named after the tasks dynamically."""
    bbh_path = "Joschka/big_bench_hard"
    local_name = "bbh"
    dataset_dir = os.path.join(BASE_DIR, local_name)
    
    # Skip if already fully downloaded
    if os.path.exists(dataset_dir) and len(os.listdir(dataset_dir)) > 0:
        print(f"\n⏭️ Skipping {local_name}: Already downloaded.")
        return
        
    print(f"\nDownloading {local_name} from {bbh_path} (Merging all tasks)...")
    
    try:
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Get all sub-task names
        tasks = get_dataset_config_names(bbh_path)
        
        # Filter out 'few_shot_prompts' as it's a metadata config, not a task
        tasks = [t for t in tasks if t != 'few_shot_prompts']
        dfs = []
        
        for task in tasks:
            # FIX: In this specific repo, the split name is identical to the task name
            ds = load_dataset(bbh_path, task, split=task)
            df = ds.to_pandas()
            df['bbh_task_category'] = task # Tag the row with its specific logic category
            dfs.append(df)
            
        combined_df = pd.concat(dfs, ignore_index=True)
        
        parquet_path = os.path.join(dataset_dir, f"{local_name}.parquet")
        jsonl_path = os.path.join(dataset_dir, f"{local_name}.jsonl")
        
        combined_df.to_parquet(parquet_path, index=False)
        combined_df.to_json(jsonl_path, orient="records", lines=True)
        
        print(f"✅ Saved {local_name} (Combined {len(combined_df)} rows) to {dataset_dir}/")
        
    except Exception as e:
        print(f"❌ Error downloading BBH: {e}")

if __name__ == "__main__":
    os.makedirs(BASE_DIR, exist_ok=True)
    pull_standard_datasets()
    pull_bbh_dataset()
    print("\n🚀 All datasets downloaded and formatted successfully!")