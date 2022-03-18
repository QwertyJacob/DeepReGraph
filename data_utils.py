import pandas as pd

def get_link_matrix(data_path='preprocessed_data/Link_Matrix_Splitted/'):
    print('Assembling link matrix...')
    # We assemble it together:
    link_ds_part_0 = pd.read_csv(data_path + 'Link_Matrix_piece_0.csv', index_col=0)
    link_ds_part_1 = pd.read_csv(data_path + 'Link_Matrix_piece_1.csv', index_col=0)
    link_ds_part_2 = pd.read_csv(data_path + 'Link_Matrix_piece_2.csv', index_col=0)
    link_ds_part_3 = pd.read_csv(data_path + 'Link_Matrix_piece_3.csv', index_col=0)
    link_ds_part_4 = pd.read_csv(data_path + 'Link_Matrix_piece_4.csv', index_col=0)
    link_ds = pd.concat([link_ds_part_0, link_ds_part_1, link_ds_part_2, link_ds_part_3, link_ds_part_4])
    return link_ds