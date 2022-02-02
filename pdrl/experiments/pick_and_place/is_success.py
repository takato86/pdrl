

def is_success(done, info):
    if not done:
        return False
    return bool(info["is_success"])
