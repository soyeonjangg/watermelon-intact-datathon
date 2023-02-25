# Team Watermelon
**Note:** Please upload the `new_train.csv` and `new_test.csv` in the `data` directory before running any of the code below. 

To create a virtual environment with all the required packages to run the model, run the following in the root directory:
```
conda create --name <env_name> --file requirements.txt
```

To reproduce the model, run
```
python extracted_data.py
python big_main.py
python small_main.py
```

The model will be saved under `data/multi-label` directory.
