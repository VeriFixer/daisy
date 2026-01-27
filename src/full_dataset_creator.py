# Get all assertions

from datasets.dafny_get_all_assertions import dafny_get_all_assertions
from datasets.dafny_dataset_generator import dafny_dataset_generator
from datasets.laurel_position_inferer import laurel_position_inferer
from datasets.dafny_dataset_all_positions_gatherer import  dafny_dataset_all_positions_gatherer

# Runs custom Dafny Fork to retrieve all assertions existent on the Dafnybench dataset
#dafny_get_all_assertions()
# Systematic removes assertions savin in new dataset when failing occurs
#dafny_dataset_generator()
# Expand dataset with extra info such as laurel fix positions, syntatic valid positions
# valid posiitons
#dafny_dataset_all_positions_gatherer()
laurel_position_inferer()