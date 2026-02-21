import torch
from torch.utils.data import random_split, Dataset, DataLoader, Subset, ConcatDataset
import pandas as pd
import random
import ast
import dask.dataframe as dd
import os
from sklearn.model_selection import train_test_split
from pytorch_lightning import LightningDataModule
from tqdm import tqdm
import gc
import psutil
import time
import copy

import esm
from esm import pretrained
from transformers import AutoTokenizer, AutoModel


########################################
# Dataset iterator with masking tokens #
########################################

class TextSeqPairing_Dataset(Dataset):

    def __init__(self, args: any, df: pd.Series):

        # dataframe
        self.df = df
        self.length = self.df.shape[0]
        self.df_column_names = self.df.columns.tolist()
        self.protein_sequence_list = self.df[args.sequence_keyword].tolist()
        self.text_captions_list = self.df['[final]text_caption'].tolist()
        self.accession_id_list = self.df[args.id_keyword].tolist()

        # parameters
        self.text_max_length = args.text_max_length # max BERT sequence tokenization length
        self.seq_max_length = 1024 # max ESM model

        # tokenizers
        self.text_tokenizer = AutoTokenizer.from_pretrained(args.text_model_path) # for text encoder
        _, self.sequence_tokenizer = pretrained.load_model_and_alphabet(args.seq_model_path) # for protein encoder

    def caption_tokenizer(self, batch_captions: list) -> dict:
        
        # transform input text tokens
        text_inputs = self.text_tokenizer.batch_encode_plus(
                            batch_captions,
                            truncation=True,
                            max_length=self.text_max_length,
                            padding='max_length',
                            return_tensors='pt',
                            return_attention_mask=True,
                            return_token_type_ids=False
        )
        
        # track the original natural language captions
        text_inputs['orig_captions'] = batch_captions

        return text_inputs
    
    def protein_tokenizer(self, batch_sequences: list) -> dict:
        
        # perpare data for ESM
        batch_converter = self.sequence_tokenizer.get_batch_converter()
        batch_labels, batch_str, batch_tokens = batch_converter(batch_sequences)
        
        # pad sequences
        batch_tokens = torch.cat((
            batch_tokens,
            torch.ones((1,1024-batch_tokens.shape[1])),
            ), dim=-1
        )

        sequence_inputs = {
            'protein_sequence_labels': batch_labels, # UniProtKB id
            'protein_sequence_str': batch_str, # original protein sequence (in amino acids)
            'protein_sequence_tokens': batch_tokens.long() # training data
        }

        return sequence_inputs
    
    
    def __getitem__(self, idx: torch.Tensor) -> (
            dict,
            dict
        ):
        
        protein_sequence = self.protein_sequence_list[idx]
        text_captions = self.text_captions_list[idx]
        accession_id = self.accession_id_list[idx]

        # prepare protein sequence in ESM format (e.g. tuple: (header, sequence)):
        batch_sequences = [
            (accession_id, protein_sequence)
        ]
        
        text_data = self.caption_tokenizer(batch_captions=[text_captions])
        protein_data = self.protein_tokenizer(batch_sequences=batch_sequences)
 
        return (
                text_data['input_ids'],
                protein_data['protein_sequence_tokens']
        )

    def __len__(self):
        return self.length


######################
# Default DataModule #
######################


class Default_DataModule(LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # construct dataset iterator
        dataset_options = {
                'default': TextSeqPairing_Dataset,
                'masked': MaskTextSeqPairing_Dataset,
                'pfam': Pfam_TextSeqPairing_Dataset,
                'pfam_ablated': Pfam_TextSeqPairing_Dataset
        }

        self.dataset_class = dataset_options.get(args.dataset_type, TextSeqPairing_Dataset)
        
    def prepare_data(self):
        pass

    def setup(self, stage=None):
        
        if self.trainer is not None:
            print(f"Number of GPUs: {self.trainer.world_size}")
            print(f"Current GPU index: {self.trainer.local_rank}")

        # Load Swiss-Prot data
        df = self.load_swiss_prot()
        
        # Split the dataframe into train and valid sets
        train_df, valid_df = train_test_split(
            df,
            test_size=self.args.valid_size,
            random_state=self.args.seed
        )
 
        print(f"Available memory after pfam_df: {check_available_memory()} GB")

        # Define datasets and dataloaders
        self.train_dataset = self.dataset_class(args=self.args, df=train_df)
        self.valid_dataset = self.dataset_class(args=self.args, df=valid_df)

    def load_swiss_prot(self) -> pd.Series:
        # Load and preprocess data (called on each GPU/TPU in DDP)
        print(f'Load Swiss-Prot data...')

        # Load Swiss-Prot data
        df = pd.read_csv(os.path.expanduser(self.args.data_path))
        df = df[df['protein_sequence'].apply(lambda seq: len(seq) <= 1022)]

        return df

    def train_dataloader(self):
        return DataLoader(
                self.train_dataset,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                shuffle=True,
                pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
                self.valid_dataset,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                pin_memory=True
        )

    def test_dataloader(self):
        # Define test dataloader if needed
        pass



################################
# Facilitator Dataset Iterator #
################################


class Facilitator_Dataset(Dataset):

    def __init__(self, args: any, dataset: dict):

        # Determine the device based on the number of GPUs
        device = 'cuda' if args.num_gpus >= 1 else 'cpu'

        # Check if text_embeddings is a list and convert to a tensor
        if isinstance(dataset['text_embedding'], list):
            # Convert list elements to tensors if they are not already
            text_emb_tensors = [torch.tensor(emb).to(device) if not isinstance(emb, torch.Tensor) else emb.to(device) for emb in dataset['text_embedding']]
            # Stack the list of tensors
            self.text_embeddings = torch.stack(text_emb_tensors)
        else:
            self.text_embeddings = dataset['text_embedding'].to(device)

        # Check if protein_embeddings is a list and convert to a tensor
        if isinstance(dataset['protein_embedding'], list):
            # Convert list elements to tensors if they are not already
            protein_emb_tensors = [torch.tensor(emb).to(device) if not isinstance(emb, torch.Tensor) else emb.to(device) for emb in dataset['protein_embedding']]
            # Stack the list of tensors
            self.protein_embeddings = torch.stack(protein_emb_tensors)
        else:
            self.protein_embeddings = dataset['protein_embedding'].to(device)


    def __getitem__(self, idx: torch.Tensor) -> (
            torch.Tensor,
            torch.Tensor
        ):


        z_t = self.text_embeddings[idx]
        z_p = self.protein_embeddings[idx] 

        return (
                z_t,
                z_p
        )


    def __len__(self):
        return len(self.text_embeddings)

###########################
# Facilitator Data Module #
###########################



class Facilitator_DataModule(LightningDataModule):
    def __init__(self, args):
        super().__init__()
        
        self.args = args
       
        self.OOD_pfam_labels = [
                'PF18369', # Polyketide synthase dimerisation element domain
                'PF04680', # Opioid growth factor receptor repeat
                'PF17988', # VEGFR-2 Transmembrane domain
                'PF12325', # TATA element modulatory factor 1 TATA binding
                'PF03272', # Putative mucin or carbohydrate-binding module
                'PF03938', # Outer membrane protein (OmpH-like)
                'PF17724', # Family of unknown function (DUF5568)
                'PF10696', # Protein of unknown function
                'PF11968', # 25S rRNA (adenine(2142)-N(1))-methyltransferase, Bmt2
                'PF04153' # NOT2/NOT3/NOT5 C-terminal
        ]
        

        # prepare embeddings
        #self.embedding_data = torch.load(args.swissprot_data_path)
        # dataset iterator
        #dataset = Facilitator_Dataset(args=args, dataset=self.embedding_data)
        # create a clone of the dataset
        #cloned_dataset = copy.deepcopy(dataset)

        # Get indices and split them
        #indices = list(range(len(dataset)))
        #train_indices, valid_indices = train_test_split(indices, test_size=args.valid_size, random_state=args.seed)
        
        # create full dataloader
        #self.all_dataloader = DataLoader(cloned_dataset, batch_size=args.batch_size, shuffle=False)
        
        # Create PyTorch DataLoader using the indices
        #self.train_sampler = Subset(dataset, train_indices)
        #self.valid_sampler = Subset(dataset, valid_indices)
        #train_dataloader = DataLoader(train_sampler, batch_size=args.batch_size, shuffle=True)
        #valid_dataloader = DataLoader(test_sampler, batch_size=args.batch_size, shuffle=False)
    
        ##########################################
        # Load Stage 1 SwissProt+Pfam Embeddings #
        ##########################################
    
        # initialize the embedding data to None
        self.swissprot_data, self.pfam_data = None, None
    
        # get both the swissprot and pfam dataset iterator in one
        if (args.swissprot_data_path != 'None') and (args.pfam_data_path != 'None'):
            print('Load both SwissProt and Pfam dataset...')
            self.train_dataset, self.valid_dataset, self.all_swiss_dataloader, self.all_pfam_dataloader = self.load_both()

        # get the swissprot dataset iterator
        elif args.pfam_data_path == 'None':
            print('Load SwissProt dataset...')
            self.train_dataset, self.valid_dataset, self.all_swiss_dataloader = self.load_swissprot()
            self.all_pfam_dataloader = None

        # get the pfam dataset iterator 
        elif args.swissprot_data_path == 'None':
            print('Load Pfam dataset...')
            self.train_dataset, self.valid_dataset, self.all_pfam_dataloader = self.load_pfam()
            self.all_swiss_dataloader = None
            


    def load_swissprot(self):

        # prepare embeddings
        self.swissprot_data = torch.load(self.args.swissprot_data_path)
        
        # dataset iterator
        swiss_dataset = Facilitator_Dataset(args=self.args, dataset=self.swissprot_data)      
        # create a clone of the dataset
        cloned_swiss_dataset = copy.deepcopy(swiss_dataset)

        # Get indices and split them
        indices = list(range(len(swiss_dataset)))
        train_indices, valid_indices = train_test_split(indices, test_size=self.args.valid_size, random_state=self.args.seed)
        
        # Create Pytorch iterator using the indices
        swiss_train_subset = Subset(swiss_dataset, train_indices)
        swiss_valid_subset = Subset(swiss_dataset, valid_indices)

        # Create Pytorch dataloader on all samples
        swiss_all_dataloader = DataLoader(cloned_swiss_dataset, batch_size=self.args.batch_size, shuffle=False)

        
        return (
                swiss_train_subset,
                swiss_valid_subset,
                swiss_all_dataloader
        )
    
    
    def load_pfam(self):

        # prepare embeddings
        self.pfam_data = torch.load(self.args.pfam_data_path)
        
        # dataset iterator
        pfam_dataset = Facilitator_Dataset(args=self.args, dataset=self.pfam_data)      
        # create a clone of the dataset
        cloned_pfam_dataset = copy.deepcopy(pfam_dataset)

        # Get indices and split them
        indices = list(range(len(pfam_dataset)))
        train_indices, valid_indices = train_test_split(indices, test_size=self.args.valid_size, random_state=self.args.seed)
        
        # Create Pytorch Dataloader using the indices
        pfam_train_subset = Subset(pfam_dataset, train_indices)
        pfam_valid_subset = Subset(pfam_dataset, valid_indices)

        # Create Pytorch dataloader on all samples
        pfam_all_dataloader = DataLoader(cloned_pfam_dataset, batch_size=self.args.batch_size, shuffle=False)

        return (
                pfam_train_subset,
                pfam_valid_subset,
                pfam_all_dataloader
        )
    

    def load_both(self):

        # get swissprot
        swissprot_train_subset, swissprot_valid_subset, swissprot_all_dataloader = self.load_swissprot()

        # get pfam
        pfam_train_subset, pfam_valid_subset, pfam_all_dataloader = self.load_pfam()
    
        # combined subsets 
        combined_train_subset = ConcatDataset([swissprot_train_subset, pfam_train_subset])
        combined_valid_subset = ConcatDataset([swissprot_valid_subset, pfam_valid_subset])

        return (
                combined_train_subset,
                combined_valid_subset,
                swissprot_all_dataloader,
                pfam_all_dataloader
        )


    def train_dataloader(self):
        return DataLoader(
                self.train_dataset,
                #self.train_sampler,
                batch_size=self.args.batch_size,
                #num_workers=self.args.num_workers,
                shuffle=True,
                #pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
                self.valid_dataset,
                #self.valid_sampler,
                batch_size=self.args.batch_size,
                #num_workers=self.args.num_workers,
                #pin_memory=True
        )

    def test_dataloader(self):
        # Define test dataloader if needed
        pass


