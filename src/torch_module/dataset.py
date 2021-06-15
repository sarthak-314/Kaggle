import torch 

class CompDataset(torch.utils.data.Dataset): 
    def __init__(self, df, feature_cols, transforms, df_type): 
        self.df = df
        self.transforms = transforms
        self.df_type = df_type
        
    def read_input(self, row): 
        return src.data.read_input(row.file_path)
    
    def __getitem__(self, idx): 
        row = self.df.iloc[idx]
        img = self.read_input(row)
        img = self.transforms(img)
        output_dict = {
            'img': torch.tensor(img, dtype=torch.float) 
        }
        if self.df_type != 'test': 
            label = row.label
            label = torch.tensor(label, dtype=torch.long)
            output_dict['label'] = label
        return output_dict
    
    def __len__(self): 
        return len(self.df)

