# Where the datasets are
#DATA_DIR = 'data'
DATA_DIR = 'training_data/nmos_FET_len_0p15_wid_1p6'
#DATASETS_JSON = DATA_DIR + '/datasets.json'
NUM_TRAINING_SETS = 10000
DATASETS_JSON = [f'{DATA_DIR}/{NUM_TRAINING_SETS}_runs_IvVg_1.48_vd_data.csv', f'{DATA_DIR}/{NUM_TRAINING_SETS}_runs_param_data.csv']



MYSTIC_MC=False

# Choose a parameter to train the model for (now we are training on all params, so just pass 'iv')
PARAM='iv'

KERAS_TUNER = True
TUNER_LIMITS = False
TUNER_DROPOUT = True
FOUR_LAYERS = False

# Enable data augmentation/scaling, etc
DATA_AUGMENTATION = True
SCALE_Y = True
LOG_DRAIN_CURRENT = True

# We use a simple fully connected network (MLP) 
# 4 layers because deeper NNs can capture more complex patterns
# Gradually decrease the neuron size to better capture patterns while avoiding overfitting
NEURONS_PER_LAYER = [1700,3900,4500]
TRAIN_DROPOUT_RATE = 0

# Training hyper-parameters

# Learning Rate gives the step size that the optimizer takes while learning, 
# smaller step size means slower convergence but more accuracy
LR_INITIAL = 9.9904e-05 

# Learning rate decay helps the model become refined as it gets closer to a minimum
# The learning rate decay steps desides how many steps the learning rate will decay after
LR_DECAY_STEPS = 100  # 100 best for log phig1 cadence data

# LR_INITIAL * LR_DECAY_RATE after each number of LR_DECAY_STEPS
LR_DECAY_RATE = 0.99

# Staircase or continuous?
LR_STAIRCASE = False

TRAIN_EARLY_STOPPING_PATIENCE = 50
TRAIN_BATCH_SIZE = 32 # 32 default
TRAIN_VALIDATION_SPLIT = 0.2

TRAIN_LOSS = 'mean_squared_error'
#TRAIN_LOSS = 'mean_squared_logarithmic_error'



