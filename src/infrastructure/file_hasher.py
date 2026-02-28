import hashlib 
from pathlib import Path 
 
 
def compute_file_hash(file_path: str) -> str: 
    """ 
    Compute SHA-256 hash of a file's contents. 
    Used to detect whether a source document has changed since last indexing. 
    """ 
    sha256 = hashlib.sha256() 
    with open(file_path, "rb") as f: 
        for chunk in iter(lambda: f.read(8192), b""): 
            sha256.update(chunk) 
    return sha256.hexdigest() 
 
 
def compute_directory_hashes(directory_path: str) -> dict[str, str]: 
    """ 
    Compute SHA-256 hashes for all supported files in a directory. 
    Returns: { filename: hash_string } 
    """ 
    supported_extensions = {".pdf", ".txt", ".md", ".csv"} 
    data_dir = Path(directory_path) 
 
    return { 
        file_path.name: compute_file_hash(str(file_path)) 
        for file_path in sorted(data_dir.rglob("*")) 
        if file_path.suffix.lower() in supported_extensions 
    }
