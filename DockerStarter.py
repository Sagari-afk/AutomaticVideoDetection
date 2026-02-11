import subprocess
import os


def start_docker():
    try:
        if check_image_existence("transnet"):
            print("Image transnet already exists, skipping build")
        else:
            print("Image transnet BUILD START")
            transnet_build = ["docker", "build", "-t", "transnet", "-f", "inference/Dockerfile", "."]
            #relativna path, ak chcem absolutnu tak import os
            #root_dir = os.path.dirname(os.path.abspaht(__file__)) toto da abs cestu k projektu
            # transnet_path = os.path.join(root_dir, "TransNetV2-master")
            transnet_path = os.path.abspath("TransNetV2-master").replace("\\", "/")
            subprocess.run(transnet_build, cwd=transnet_path, check=True)
            print("Image transnet BUILD END")

        if check_image_existence("action-recognition"):
            print("Image action-recognition already exists, skipping")
        else:
            print("Image action-recognition BUILD START")
            action_recognition_build = ["docker", "build", "-t", "action-recognition", "-f", "Dockerfile", "."]
            action_recognition_path = os.path.abspath("ActionRecognition").replace("\\", "/")
            subprocess.run(action_recognition_build, cwd=action_recognition_path, check=True)
            print("Image action-recognition BUILD END")
        return
    except subprocess.CalledProcessError as e:
        print(f"Docker builds failed due to exception: {e}")
        raise e
    except FileNotFoundError as e:
        print(f"Docker not installed or not found due to exception: {e}")
        raise e


def check_image_existence(image_name: str) -> bool:
    try:
        print(f"Checking if {image_name} exists BEGIN")
        response = subprocess.run(["docker", "images", "-q", image_name], capture_output=True, text=True)
        result = bool(response.stdout.strip())
        print(f"Checking if {image_name} exists END. Result: {result}")
        return result
    except FileNotFoundError as e:
        print(f"Docker not installed or not found due to exception: {e}")
        raise e


# if __name__ == "__main__":
#     start_docker()
