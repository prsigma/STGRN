from torch.utils.data import Dataset
import warnings
warnings.filterwarnings('ignore')

class STGRN_data(Dataset):
    def __init__(self, data, dropout_mask, flag, size):
        self.seq_len = size[0]
        self.pred_len = size[1]
        
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.data = data
        self.mask = dropout_mask
        
        self.__read_data__()

    def __read_data__(self):
        num_train = int(len(self.data) * 0.7)
        
        border1s = [0, num_train - self.seq_len, 0]
        border2s = [num_train, len(self.data), len(self.data)]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        self.data_x = self.data[border1:border2]
        self.data_y = self.data[border1:border2]
        
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_y_mask = self.mask[r_begin:r_end]

        return seq_x, seq_y, seq_y_mask

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1
