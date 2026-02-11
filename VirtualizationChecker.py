import os
import subprocess as sp

#for windows
#can be used to check if virtualization is enabled on windows


def check_enabled_virtualization():
    try:
        sys_info = sp.run(['systeminfo'], capture_output=True, text=True)
        if "Virtualization Enabled In Firmware" in sys_info:
            print("Virtualization ENABLED")
            return True
        else:
            print("Virtualization DISABLED")
            return False
    except Exception as e:
        print("Error during virtualization check: ", e)
        return False

