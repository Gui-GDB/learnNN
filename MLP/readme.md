1. Download the dataset from https://www.kaggle.com/datasets/fusicfenta/cat-and-dog?resource=download and organize the directory as follows:

```plain-text
 └─data
    └─dog_and_cat
        ├─single_prediction
        ├─test_set
        │  ├─cats
        │  └─dogs
        └─training_set
            ├─cats
            └─dogs
```
   
2. Modify the path in "main" scripts to select the specified dataset:

   - MNIST

   - dog_and_cat

       ```python
       parser.add_argument('--dataset', type=str, default='dog_and_cat', help='Dataset name.')
       ```

   - Modify the default parameters with the dataset you want to select.

3. Run `main.py`.