def log_info(msg: str) -> None:
    print(f"[INFO] {msg}")

def log_success(msg: str) -> None:
    print(f"[SUCCESS] {msg}")

def log_error(msg: str) -> None:
    print(f"[ERROR] {msg}")

def log_warning(msg: str) -> None:
    print(f"[WARNING] {msg}")

def print_header(title: str, width: int = 60) -> None:
    print("=" * width)
    print(title)
    print("=" * width)

def print_separator(width: int = 60) -> None:
    print("-" * width)
