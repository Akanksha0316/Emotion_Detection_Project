import zipfile
import json
import os
import shutil


# Original model
original_model = "model/emotion_detection_bilstm.keras"

# Fixed model
fixed_model = "model/emotion_detection_bilstm_fixed.keras"

# Temporary folder
temp_folder = "model/temp_model"


print("Starting model fix...")


# --------------------------------------------------
# 1. Check original model
# --------------------------------------------------

if not os.path.exists(original_model):
    print("ERROR: Original model file was not found.")
    exit()

print("Original model found.")


# --------------------------------------------------
# 2. Remove old temporary folder if it exists
# --------------------------------------------------

if os.path.exists(temp_folder):
    shutil.rmtree(temp_folder)

os.makedirs(temp_folder)


# --------------------------------------------------
# 3. Extract the .keras file
# --------------------------------------------------

print("Extracting model...")

with zipfile.ZipFile(original_model, "r") as zip_ref:
    zip_ref.extractall(temp_folder)


# --------------------------------------------------
# 4. Open config.json
# --------------------------------------------------

config_file = os.path.join(temp_folder, "config.json")

if not os.path.exists(config_file):
    print("ERROR: config.json was not found inside the model.")
    shutil.rmtree(temp_folder)
    exit()

print("Reading model configuration...")


with open(config_file, "r", encoding="utf-8") as file:
    config = json.load(file)


# --------------------------------------------------
# 5. Remove incompatible quantization_config entries
# --------------------------------------------------

removed_count = 0


def remove_quantization_config(obj):
    global removed_count

    if isinstance(obj, dict):

        if "quantization_config" in obj:
            if obj["quantization_config"] is None:
                del obj["quantization_config"]
                removed_count += 1

        for value in list(obj.values()):
            remove_quantization_config(value)

    elif isinstance(obj, list):

        for item in obj:
            remove_quantization_config(item)


remove_quantization_config(config)


# --------------------------------------------------
# 6. Save modified configuration
# --------------------------------------------------

with open(config_file, "w", encoding="utf-8") as file:
    json.dump(config, file, indent=2)


print("Removed incompatible entries:", removed_count)


# --------------------------------------------------
# 7. Create new .keras file
# --------------------------------------------------

print("Creating fixed model...")


with zipfile.ZipFile(
    fixed_model,
    "w",
    compression=zipfile.ZIP_DEFLATED
) as zip_ref:

    for root, dirs, files in os.walk(temp_folder):

        for file in files:

            file_path = os.path.join(root, file)

            archive_path = os.path.relpath(
                file_path,
                temp_folder
            )

            zip_ref.write(
                file_path,
                archive_path
            )


# --------------------------------------------------
# 8. Remove temporary folder
# --------------------------------------------------

shutil.rmtree(temp_folder)


print()
print("========================================")
print("MODEL FIX COMPLETED")
print("========================================")
print()
print("Fixed model:")
print(fixed_model)
print()
print("Original model was NOT changed.")
print("Backup is still safe.")