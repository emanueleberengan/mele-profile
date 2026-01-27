from gpt_download import download_and_load_gpt2

settings, params = download_and_load_gpt2(
    model_size="124M", models_dir="gpt2"
)

# inspect the code of downloaded model from openAi
print("Settings: ", settings)
print("Parameter dictonary keys: ", params.keys())