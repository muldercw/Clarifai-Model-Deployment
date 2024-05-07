from clarifai.client import App
from utils.config_processor import update_config
from utils.tests.run_inference import run_tests
import os, re, time
import docker
import sys


def generatedockerimagename(app_id):
    app_id = re.sub(r'\W+', '', app_id)
    return f"triton_{app_id}".lower()

def export_mod(user_id, pat, app_id):
    app = App(app_id=app_id, pat=pat)
    for model_ in app.list_models(only_in_app=True):
        try:
            modelname = model_.name
            model_.export(export_dir="./models")
            unzip_tar_files(modelname)
        except Exception as e:
            print(f"Error exporting model {model_.id}: {e}")

def unzip_tar_files(modelname):
    tarfiles = [f for f in os.listdir(f'./models/') if f.endswith('.tar')]
    for tarfile in tarfiles:

        os.makedirs(f'./models/{modelname}', exist_ok=True)
        os.system(f"tar -xvf ./models/{tarfile} -C ./models/{modelname}")
        os.remove(f'./models/{tarfile}')

def prep_models():
    model_directory = './models' 
    folders = [f for f in os.listdir(model_directory) if os.path.isdir(os.path.join(model_directory, f))]
    for folder in folders:
        modeldirectory = f'./models/{folder}'
        update_config(modeldirectory)

def build_docker(imagename):
    print("Building docker image...")
    client = docker.from_env(timeout=120)
    image, _ = client.images.build(path='.', tag=imagename, dockerfile='Dockerfile', rm=True)
    print("Docker image built successfully! Saving image...")
    os.system(f"docker save {imagename} -o {imagename}.tar")
    print("Docker image saved successfully!")
    time.sleep(3)


def run_docker(imagename):
    print("Running docker image...")
    client = docker.from_env()
    current_directory = os.getcwd()
    client.containers.run(imagename, detach=True, shm_size='3G', device_requests=[docker.types.DeviceRequest(device_ids=["0"], capabilities=[['gpu']])], 
                      ports={'8000': '8000', '8001': '8001'}, name=imagename, volumes={f'{current_directory}/models': {'bind': '/models', 'mode': 'rw'}})


def pipeline():
    user_id = sys.argv[1]
    pat = sys.argv[2]
    app_id = sys.argv[3]
    imagename = generatedockerimagename(app_id)
    os.makedirs(f'./models', exist_ok=True)
    export_mod(user_id, pat, app_id)
    prep_models()
    build_docker(imagename)
    run_docker(imagename)
    run_tests()
    print("Pipeline completed successfully!")


if __name__ == '__main__':
    pipeline()
    
