import os
from collections import Counter
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd


class VocalSetDataset(Dataset):

    def __init__(self,
                 directory_path,
                 measurements_filename,
                 device,
                 normtype = 'mean',
                 audio_embedding_model = None):
        self.device = device
        print(f'Using {device} as device for dataset.')
        self.dir_path = directory_path
        self.measurements_filename = measurements_filename
        self.vq_measurements = self._list_of_vq_measurements()
        self.label_names = ['id', 'gender', 'phrase','technique','vowel']
        self.df_names = [] #DataFrame consisting of 1 column, assigned in dataframe load
        self.normtype = normtype
        self.feat_stats = {}
        self.df = self._load_dataframe()
        if audio_embedding_model is not None:
            self.audio_embedding_model = audio_embedding_model
            self.embed_audio = True
            self.audio_embeddings = self._create_audio_embeddings()
        else:
            self.embed_audio = False


    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        name = self._get_sample_name(index)
        measurements = self._get_sample_measurements(index).to(self.device)
        labels = self._get_sample_labels(name)
        if self.embed_audio:
            embedding = self.audio_embeddings[index]
            return name, measurements, labels, embedding
        else:
            return name, measurements, labels

    def _get_data_frame(self):
        return self.df

    def _load_dataframe(self):
        df = self._read_measurements_file()
        assert list(df.columns) == self.vq_measurements
        self.vq_measurements = self.vq_measurements[1:] #chop off filename
        #normalize features
        df = self._convert_to_float(df)
        self.df_names = df['Input File']
        df = self._normalize_features(df)
        return df

    def _read_measurements_file(self):
        #read in measurements excel file produced by VoiceLab
        xls = pd.ExcelFile(os.path.join(self.dir_path, self.measurements_filename))
        df = pd.read_excel(xls, 'Summary') #all measurements in Summary sheet
        df = df[df.columns.intersection(self.vq_measurements)]
#         df.drop(list(df.filter(regex = 'Input File.')), axis = 1, inplace = True) #removes duplicate columns listing filename
        return df
    
    def _convert_to_float(self,df):
        # reformat to have all valid values in measurement cols
        for i,col in enumerate(df.columns):
            if i==0:
                pass
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        if self.normtype == 'mean':
            df = df.replace(np.nan, 0, regex=True)
        else:
            df = df.replace(np.nan, 0.5, regex=True) #minmax normalization
        return df

    def _normalize_features(self, df):
        # apply minmax-normalization to columns
        df = df.iloc[:,1:] #drop filename col
        self.feat_stats['mean'] = df.mean()
        self.feat_stats['std'] = df.std()
        self.feat_stats['min'] = df.min()
        self.feat_stats['max'] = df.max()
        if self.normtype=='mean': #default is minmax
            print('Applying mean-normalization to all features')
            normalized_df=(df-self.feat_stats['mean'])/self.feat_stats['std']
        else:
            print('Applying minmax-normalization to all features')
            normalized_df=(df-self.feat_stats['min'])/(self.feat_stats['max']-self.feat_stats['min'])
        return normalized_df

    def _unnormalize_features(self):
        if self.normtype == 'mean':
            df = self.df*self.feat_stats['std'] + self.feat_stats['mean']
        elif self.normtype == 'minmax':
            df = self.df*(self.feat_stats['max'] - self.feat_stats['min']) + self.feat_stats['min']
        return df

    def _get_sample_name(self, index):
        return self.df_names.iloc[index] #index of row

    def _get_sample_measurements(self, index):
        row = self.df.iloc[index].tolist()
        return torch.tensor(row)

    def _embed_audio(self, name):
        filepath = os.path.join(self.dir_path, f'{name}.wav')
        return self.audio_embedding_model.extract_features(filepath)
    
    def _create_audio_embeddings(self):
        print('Creating audio embeddings using given model...')
        return [self._embed_audio(name) for name in self.df_names]

    # The following methods create Corresponding Labels from VocalSet Filename Metadata
    def _removeItem(self,item, labels):
        if item in labels: labels.remove(item)
        return labels

    def _replaceItem(self,item1,item2,labels):
        if(item1 in labels):
            labels[labels.index(item1)] = item2
        return labels

    def _combine(self,item1, item2, labels):
        if((item1 in labels) and (item2 in labels)):
            combo = item1+item2
            labels[labels.index(item1)] = combo
            labels.remove(item2)
        return labels

    def _splitItem1(self,labels):
        labels.insert(0,labels[0][0])
        labels[1] = labels[1][1:]
        return labels

    def _cleanUpLabels(self,labels):
        labels = self._removeItem('c',labels)
        labels = self._removeItem('f',labels) #do before gender, singer separation!
        labels = self._replaceItem('u(1)','u',labels)
        labels = self._replaceItem('a(1)','a',labels)
        labels = self._replaceItem('arepggios','arpeggios',labels)
        labels = self._combine('fast','forte',labels)
        labels = self._combine('fast','piano',labels)
        labels = self._combine('slow','piano',labels)
        labels = self._combine('slow','forte',labels)
        labels = self._combine('lip','trill',labels)
        labels = self._combine('vocal','fry',labels)
        labels = self._splitItem1(labels)
        return labels

    def _parseLabels(self,filename):
        # info,ext = os.path.splitext(filename)
        # if ext=='.csv':
        #     info = os.path.splitext(info)[0] #remove crepe .f0 tag
        filename = filename[:-3]
        lbl = filename.split("_") #known delimiter
        return self._cleanUpLabels(lbl)

    def _get_sample_labels(self,name):
        lbls = self._parseLabels(name)
        #print(lbls)
        gender = lbls[0]
        singer = lbls[1]
        phrase = lbls[2]
        technique = lbls[3]
        try:
            vowel = lbls[4]
        except:
            vowel = 'N'
#         print("labels generated")
        return gender, singer, phrase, technique, vowel
    
#     def _get_num_unique_labels(self):
#         if len(self.labels) == 0:
#         for i in range(len(self.directory)):
#             self.labels.append(self._get_sample_labels(i))
#         labels = Counter(self.labels).keys()
#         counts = Counter(self.labels).values()
#         return labels, counts

    # Measurements of interest specific to voice quality / timbre
    def _list_of_vq_measurements(self):
        return ['Input File',
                 'subharmonic-to-harmonic ratio',
                 'Subharmonic Mean Pitch',
                 'Harmonics to Noise Ratio',
                 'Local Jitter',
                 #'Local Absolute Jitter',
                 #'RAP Jitter',
                 #'ppq5 Jitter',
                 #'ddp Jitter', #'PCA Result', <- causing error
                 #'local_shimmer',
                 'localdb_shimmer',
                 #'apq3_shimmer',
                 #'aqpq5_shimmer',
                 #'apq11_shimmer',
                 #'dda_shimmer',
                 #'PCA Result.1',
                 'cpp',
                 'Spectral Tilt',
                 'Centre of Gravity',
                 'Standard Deviation',
                 #'Kurtosis',
                 'Skewness',
                 'Band Energy Difference',
                 'Band Density Difference']