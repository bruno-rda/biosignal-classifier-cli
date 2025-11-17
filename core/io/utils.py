from pathlib import Path
import json

def get_next_dir(base_dir: str, prefix: str) -> str:
    base = Path(base_dir)
    existing = [
        d.name for d in base.iterdir() 
        if d.is_dir() and d.name.startswith(prefix)
    ]
    
    numbers = [
        int(int_name) for name in existing
        if (int_name := name.replace(prefix + '_', '')).isdigit()
    ]
    
    next_num = max(numbers, default=0) + 1
    dir_path = base / f'{prefix}_{next_num:03d}'
    dir_path.mkdir(parents=True, exist_ok=True)
    return str(dir_path)

def save_json(data: dict, path: str):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4, default=str)

def load_json(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)