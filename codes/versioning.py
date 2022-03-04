###############################
#                             
#       SAVE CSV VERSION
#                             
###############################

from os import path
import pandas as pd

def save_csv_version(file_path, 
                     df, 
                     min_version = 1, 
                     **args):
    
    '''
    Saves pandas DF as a csv file with an automatically assigned version number 
    to prevent overwriting the existing file. If no file with the same name 
    exists in the specified path, '_v1' is appended to the file name to indicate 
    the version of the saved data. If such a version already exists, the function 
    iterates over integers and saves the data as '_v[k]', where [k] stands for 
    the next available integer. 

    --------------------
    Arguments:
    - file_path (str): file path including the file name
    - df (pandas DF): dataset
    - min_version (int): minimum version number
    - **args: further arguments to pass to pd.to_csv() function

    --------------------
    Returns:
    - None

    --------------------
    Examples:
    
    # import dependencies
    import pandas as pd
    import numpy as np

    # create data frame
    data = {'age': [27, np.nan, 30, 25, np.nan], 
        'height': [170, 168, 173, 177, 165], 
        'gender': ['female', 'male', np.nan, 'male', 'female']}
    df = pd.DataFrame(data)

    # first call saves df as 'data_v1.csv'
    save_csv_version('data.csv', df, index = False)

    # second call saves df as 'data_v2.csv' as data_v1.csv already exists
    save_csv_version('data.csv', df, index = False)
    '''
    
    # tests
    assert isinstance(df, pd.DataFrame), 'df has to be a pandas dataframe'

    # initialize
    version = min_version - 1
    is_version_present = True

    # update name
    file_path_version = file_path.replace('.csv', ('_v' + str(version) + '.csv'))
    
    # export loop
    while is_version_present:

        # update file name
        version += 1
        file_path_version = file_path.replace('.csv', ('_v' + str(version) + '.csv'))

        # check for a file with the same name
        is_version_present = path.isfile(file_path_version)

    # save file
    df.to_csv(file_path_version, **args)
    print('Saved as ' + file_path_version)
