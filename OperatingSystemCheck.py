from sys import platform



def operating_system_check():

    if platform == "linux" or platform == "linux2":
        return "Linux"
    elif platform == "darwin":
        return "MacOS"
    elif platform == "win32":
        return "Windows"
    elif platform in ["aix", "android", "ios", "emscripten", "cygwin", "wasi"]:
        return "Invalid"
    else:
        return "Unknown"


