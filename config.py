##########################
#      model config      #
##########################

seed = 3
learning_rate = 0.001
dropout_rate = 0.1
batch_size = 256
num_epochs = 200
encoding_size = 16
hidden_units = [256, 256]
buffer_epochs = 30

##########################
#   grid search config   #
##########################

TARGET_FEATURE_NAMEs = ['DO', "Chl_a"]
STEADY_FEATURE_NAMEs = ['Lat', 'Long']
VARIABLE_FEATURE_NAMEs = ['Depth', 'pH', 'Temp', 'Density']
ONE_OF_FEATURE_NAMEs = ['Sal', 'Cond', 'EC25']
MODEL_LIST = ['variable_selection']
simple_grid = True
grid_search_result_path = 'results/simple_grid_result.xlsx'

##########################
#     dataset config     #
##########################

MAPPING_NAME = {'Depth [m]': 'Depth',
                'Temp. [degC]': 'Temp',
                'Sal.': 'Sal',
                'Cond. [mS/cm]': 'Cond',
                'EC25 [µS/cm]': 'EC25',
                'Density [kg/m3]': 'Density',
                'Chl-Flu. [ppb]': 'Chl_Flu',
                'Chl-a [µg/l]': 'Chl_a',
                'Turb. [FTU]': 'Turb',
                'ORP [mV]': 'ORP',
                'DO [%]': 'DO_p',
                'DO [mg/l]': 'DO',
                'Quant. [µmol/(m2*s)]': 'Quant'}

all_features = ['Lat', 'Long', 'Depth', 'Temp',
                'pH', 'Sal', 'Cond', 'EC25', 'Density',
                'Chl_Flu', 'Chl_a', 'Turb', 'ORP', 'DO', 'Quant']
normalize = False

##########################
#     report config      #
##########################

models_name = ['base_line', 'wide_deep', 'deep_cross', 'variable_selection']
data_path = 'data/all_data.xlsx'
feature_list = ['Lat', 'Long', 'Depth', 'Temp', 'pH']
target = ['DO']
model_name = 'base_line'
shap_sample_num = 500
variation_plot_name = 'DO'
dem_path = 'data/SRTM.xyz'
variation_color_map = 'gist_rainbow'
