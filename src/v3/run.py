##mayuresh: Entry point for all functionalities in v3, to be run from folder src

import v3.feature_generation as fg
import utils.myconfig

config_file=r'/export/da/mkunjir/LabelDebugger/config/tools.config'
params = utils.myconfig.read_config(config_file)

fg.generate_features(params)
