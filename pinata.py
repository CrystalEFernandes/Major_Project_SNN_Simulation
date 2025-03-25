import requests
import json
from google.colab import files

uploaded = files.upload()
file_name = list(uploaded.keys())[0]

PINATA_API_KEY = "897bc0dbc9c242fe0c35"
PINATA_SECRET_API_KEY = "0c7ce37e46f40b32b70eee0f85ffcb43af2704dcb2af60e88d3d81e70c9fb436"
PINATA_URL = "https://api.pinata.cloud/pinning/pinFileToIPFS"

def upload_to_pinata(file_name):
    headers = {
        "pinata_api_key": PINATA_API_KEY,
        "pinata_secret_api_key": PINATA_SECRET_API_KEY
    }
    with open(file_name, "rb") as file:
        response = requests.post(PINATA_URL, files={"file": file}, headers=headers)

    if response.status_code == 200:
        data = response.json()
        print("File uploaded successfully!")
        print("IPFS Hash:", data["IpfsHash"])
        return data
    else:
        print("Error uploading file:", response.text)
        return None

file_data = upload_to_pinata(file_name)

if file_data:
    print("\nUploaded File Details:")
    print(json.dumps(file_data, indent=4))
    print("\nAccess it via IPFS Gateway:")
    print(f"https://gateway.pinata.cloud/ipfs/{file_data['IpfsHash']}")