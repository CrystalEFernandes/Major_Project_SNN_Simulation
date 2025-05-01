import requests
import json
from google.colab import files

uploaded = files.upload()
file_name = list(uploaded.keys())[0]

PINATA_API_KEY = "a60542a4fd0e913b0d0e"
PINATA_SECRET_API_KEY = "acf65be10c0c72bf313198ff93ec714c0303210659350be7df6355e064422cc3"
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