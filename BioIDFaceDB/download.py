import wget
import zipfile
import os
import shutil


database_url = "https://www.bioid.com/uploads/BioID-FaceDatabase-V1.2.zip"
extract_folder = "../datasets/BioID-FaceDatabase"
file_name = "BioID-FaceDatabase-V1.2.zip"


print("Downloading database...")
wget.download(database_url, file_name)

if os.path.exists(extract_folder):
  print("Removing existing database...")
  shutil.rmtree(extract_folder)
os.makedirs(extract_folder)

print("Extracting database...")
with zipfile.ZipFile(file_name, 'r') as zip_ref:
  zip_ref.extractall(extract_folder)

print("Dondloading complete!")
print("Cleaning up...")
os.remove(file_name)