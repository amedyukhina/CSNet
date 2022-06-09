import os


def get_paired_file_list(dir1, dir2):
    files = os.listdir(dir1)
    files1 = [os.path.join(dir1, fn) for fn in files
              if fn in os.listdir(dir2)]
    files2 = [os.path.join(dir2, fn) for fn in files
              if fn in os.listdir(dir2)]
    return files1, files2
