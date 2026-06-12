import os

def print_tree(startpath, exclude_dirs):
    for root, dirs, files in os.walk(startpath):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        level = root.replace(startpath, '').count(os.sep)
        indent = '|   ' * (level)
        print('{}{}/'.format(indent, os.path.basename(root) or '.'))
        subindent = '|   ' * (level + 1)
        for f in files:
            if not f.endswith('.pyc'):
                print('{}{}'.format(subindent, f))

print_tree('.', ['.git', '.venv', '__pycache__', '.pytest_cache', '.agent', '.idea', 'get_tree.py'])
