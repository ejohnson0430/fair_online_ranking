# Online Post-Processing In Rankings For Fair Utility Maximization


## Train a Model

To train a model on one of the provided datasets, run the train_model.py script with the following parameters:

```bash
python train_model.py <dataset> <epochs> <beta>
```

Where 'dataset' is one of the following keys: synthe, german, resume, airbnb,  or stacke, 'epochs' is the number of epochs to train for, and 'beta' is the roll-out mixture parameter (1 = always reference policy, 0 = always learned policy, or anything in between).

The script will train an L2SQ model for the specified number of epochs and save the model which scores best on validation data to the models directory.

## Plot Results

Pretrained models for each dataset are included, and you can use these to recreate the plots from the paper by running the plot_results.py script:

```bash
python plot_results.py
```