import hashlib

def file_hash(file_path):

    hasher = hashlib.md5()

    with open(file_path, "rb") as file:
        while chunk := file.read(8192):
            hasher.update(chunk)
    
    return hasher.hexdigest()