from huggingface_hub import list_repo_files, hf_hub_download

repo_id = "rootstrap-org/crowd-counting"

print("Files in repo:")
files = list_repo_files(repo_id)
print(files)

# Try common names
possible = ["pytorch_model.bin", "model.pth", "weights.pth", "csrnet.pth"]

found = None
for f in possible:
    if f in files:
        found = f
        break

if found is None:
    raise ValueError("Could not find checkpoint. Check the repo files printed above.")

print("Downloading:", found)
path = hf_hub_download(repo_id=repo_id, filename=found)
print("Saved to:", path)
