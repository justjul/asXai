import psutil


def merge_dicts(dict_list):
    merged = {}
    for d in dict_list:
        for key, value in d.items():
            merged.setdefault(key, []).append(value)  # Collect values as lists
    return merged

def close_all_chrome_sessions(verbose: bool = True):
    nkill = 0
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        if proc.info['name'] and ('chrome' in proc.info['name'].lower() or 
                                  'chromium' in proc.info['name'].lower()):
            try:
                proc.kill()
                nkill += 1
            except Exception as e:
                print(f"Could not close process {proc.info['pid']}: {e}")

    if verbose:
        print(f"Closed {nkill} Chrome processes")

