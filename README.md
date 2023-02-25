# Team Watermelon

To create a virtual environment with all the required packages to run the model, run the following in the root directory:
```
conda create --name <env_name> --file requirements.txt
```

To reproduce the model, run
```
python extracted_data.py
python main.py
```

The model will be saved under `data/multi-label` directory.