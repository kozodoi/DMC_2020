###############################
#                             
#      PRINT FACTOR LEVELS
#                             
###############################

import pandas as pd

def print_factor_levels(df, 
                        top = 5):
    
    '''
    Prints levels of categorical features in the dataset.
    
    --------------------
    Arguments:
    - df (pandas DF): dataset
    - top (int): how many most frequent values to display

    --------------------
    Returns
    - None

    --------------------
    Examples:

    # import dependencies
    import pandas as pd
    import numpy as np

    # create data frame
    data = {'age': [27, np.nan, 30, 25, np.nan], 
        'height': [170, 168, 173, 177, 165], 
        'gender': ['female', 'male', np.nan, 'male', 'female'],
        'income': ['high', 'medium', 'low', 'low', 'no income']}
    df = pd.DataFrame(data)

    # print factor levels
    print_factors(df, top = 3)
    '''
    
    # tests
    assert isinstance(df,  pd.DataFrame), 'df has to be a pandas dataframe'
    assert isinstance(top, int),          'top has to be a positive integer'

    # find factors
    facs = [f for f in df.columns if df[f].dtype == 'object']
    
    # print results
    if len(facs) > 0:
        print('Found {} categorical features.'.format(len(facs)))
        print('')
        for fac in facs:
            print('-' * 30)
            print(fac + ': ' + str(df[fac].nunique()) + ' unique values')
            print('-' * 30)
            print(df[fac].value_counts(normalize = True, dropna = False).head(top))
            print('-' * 30)
            print('')
    else:
        print('Found no categorical features.')



###############################
#                             
#     FIND CONSTANT FEATURES
#                             
###############################

import pandas as pd

def find_constant_features(df, 
                           dropna = False):
    
    '''
    Finds features that have just a single unique value.

    --------------------
    Arguments:
    - df (pandas DF): dataset
    - dropna (bool): whether to treat NA as a unique value

    --------------------
    Returns:
    - list of constant features

    --------------------
    Examples:

    # import dependencies
    import pandas as pd
    import numpy as np

    # create data frame
    data = {'age': [27, np.nan, 30, 25, np.nan], 
        'height': [170, 168, 173, 177, 165], 
        'gender': ['female', 'female', 'female', 'female', 'female']}
    df = pd.DataFrame(data)

    # check constant features
    find_constant_features(df)
    '''
    
    # tests
    assert isinstance(df, pd.DataFrame), 'df has to be a pandas dataframe'
    
    # find constant features
    constant = df.nunique(dropna = dropna) == 1
    features = list(df.columns[constant])

    # return results
    if len(features) > 0:
        print('Found {} constant features.'.format(len(features)))
        return features 
    else:
        print('No constant features found.')
        
        

###############################
#                             
#        COUNT MISSINGS       
#                             
###############################

import pandas as pd

def print_missings(df):
    
    '''
    Counts missing values in a dataframe and prints the results.

    --------------------
    Arguments:
    - df (pandas DF): dataset

    --------------------
    Returns:
    - pandas DF with missing values

    --------------------
    Examples:

    # import dependencies
    import pandas as pd
    import numpy as np

    # create data frame
    data = {'age': [27, np.nan, 30, 25, np.nan], 
        'height': [170, 168, 173, 177, 165], 
        'gender': ['female', 'male', np.nan, 'male', 'female'],
        'income': ['high', 'medium', 'low', 'low', 'no income']}
    df = pd.DataFrame(data)

    # count missings
    print_missings(df)
    '''

    # tests
    assert isinstance(df, pd.DataFrame), 'df has to be a pandas dataframe'

    # count missing values
    total = df.isnull().sum().sort_values(ascending = False)
    percent = (df.isnull().sum() / df.isnull().count()).sort_values(ascending = False)
    table = pd.concat([total, percent], axis = 1, keys = ['Total', 'Percent'])
    table = table[table['Total'] > 0]

    # return results
    if len(table) > 0:
        print('Found {} features with missing values.'.format(len(table)))
        return table 
    else:
        print('No missing values found.')
        
        
        
###############################
#                             
#     SPLIT NESTED FEATURES
#                             
###############################

import pandas as pd

def split_nested_features(df, 
                          split_vars, 
                          sep,
                          drop = True):
    
    '''
    Splits a nested string column into multiple features using a specified 
    separator and appends the creates features to the data frame.

    --------------------
    Arguments:
    - df (pandas DF): dataset
    - split_vars (list): list of string features to be split
    - sep (str): separator to split features
    - drop (bool): whether to drop the original features after split

    --------------------
    Returns:
    - pandas DF with new features

    --------------------
    Examples:

    # import dependencies
    import pandas as pd
    import numpy as np

    # create data frame
    data = {'age': [27, np.nan, 30, 25, np.nan], 
        'height': [170, 168, 173, 177, 165], 
        'income': ['high,100', 'medium,50', 'low,25', 'low,28', 'no income,0']}
    df = pd.DataFrame(data)

    # split nested features
    df_new = split_nested_features(df, split_vars = 'income', sep = ',')
    '''
    
    # tests
    assert isinstance(df, pd.DataFrame), 'df has to be a pandas dataframe'

    # copy df
    df_new = df.copy()

    # store no. features
    n_feats = df_new.shape[1]

    # convert to list
    if not isinstance(split_vars, list):
        split_vars = [split_vars]

    # feature engineering loop
    for split_var in split_vars:
        
        # count maximum values
        max_values = int(df_new[split_var].str.count(sep).max() + 1)
        new_vars = [split_var + '_' + str(val) for val in range(max_values)]
        
        # remove original feature
        if drop:
            cols_without_split = [col for col in df_new.columns if col not in split_var]
        else:
            cols_without_split = [col for col in df_new.columns]
            
        # split feature
        df_new = pd.concat([df_new[cols_without_split], df_new[split_var].str.split(sep, expand = True)], axis = 1)
        df_new.columns = cols_without_split + new_vars
        
    # return results
    print('Added {} split-based features.'.format(df_new.shape[1] - n_feats + int(drop) * len(split_vars)))
    return df_new