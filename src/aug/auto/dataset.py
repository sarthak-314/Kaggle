


class CompDatasetTest(torch.utils.data.Dataset): 
    def __init__(self, df):
        self.df = df
    
    def __getitem__(self, idx): 
        row = self.df.iloc[idx]
        
        # read the main input 
        img = read_input_file(row.file_path)
        img = self.transforms(img)
        
        # build all the features for the input
        feature_dict = {
            'img': torch.tensor(img, dtype=torch.float)
        }
        # no labels for test
        output_dict = feature_dict
        return output_dict
    
    def __len__(self): 
        return len(self.df)    

# Jupyter Testing
train, test = read_dataframes()['train'], read_dataframes()['test']
train_ds = CompDatasetTrain(train, train_transforms_comp)
test_ds = CompDatasetTest(test)