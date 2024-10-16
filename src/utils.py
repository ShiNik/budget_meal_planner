from pathlib import Path

def list_files_in_folder(folder_path: Path, file_extension: str) -> list[Path]:
    return [path for path in folder_path.glob(file_extension)]

def get_name_from_path(file_path: Path) -> str:
    return file_path.stem