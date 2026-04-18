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
import re

import esm
from esm import pretrained
from transformers import AutoTokenizer, AutoModel

from biom3.backend.device import setup_logger

logger = setup_logger(__name__)


# --- Per-batch stochastic annotation dropout for PenCL retraining ---

_FIELD_PREFIXES = [
    "PROTEIN NAME", "FUNCTION", "CATALYTIC ACTIVITY",
    "BIOPHYSICOCHEMICAL PROPERTIES", "LINEAGE", "FAMILY NAMES",
    "FAMILY NAME", "PARALOG NAME", "PARALOG FUNCTION", "SUBUNIT",
    "SUBCELLULAR LOCATION", "SIMILARITY", "DOMAIN",
    "ACTIVITY REGULATION", "PTM", "TISSUE SPECIFICITY",
    "MISCELLANEOUS", "COFACTOR", "PATHWAY", "BIOTECHNOLOGY",
    "INDUCTION",
]

_RETENTION_PROBS = {
    "PROTEIN NAME": 1.00,
    "FUNCTION": 0.85, "LINEAGE": 0.85,
    "PARALOG NAME": 0.65, "CATALYTIC ACTIVITY": 0.65,
    "FAMILY NAMES": 0.65, "FAMILY NAME": 0.65,
    "PATHWAY": 0.55, "DOMAIN": 0.55,
}
# All others default to 0.40

_FIELD_RE = re.compile(
    r'(?:^|\.\s+)(' + '|'.join(re.escape(p) for p in sorted(_FIELD_PREFIXES, key=len, reverse=True)) + r'):\s*',
    re.IGNORECASE
)


def apply_field_dropout(text, rng=None):
    """Randomly drop annotation fields from structured text.
    For NL text (no field prefixes), applies sentence-level dropout."""
    if rng is None:
        rng = random

    # If text doesn't contain field prefixes, apply sentence dropout
    if not any(f"{prefix}:" in text for prefix in _FIELD_PREFIXES[:3]):
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        kept = [s for s in sentences if rng.random() < 0.7]
        return '. '.join(kept) + '.' if kept else text

    # Field-level dropout for structured text
    matches = list(_FIELD_RE.finditer(text))
    if not matches:
        return text

    fields = []
    for i, m in enumerate(matches):
        name = m.group(1).upper()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip().rstrip('.')
        if content:
            fields.append((name, content))

    kept = []
    for name, content in fields:
        prob = _RETENTION_PROBS.get(name, 0.40)
        if rng.random() < prob:
            kept.append(f"{name}: {content}")

    result = '. '.join(kept) + '.' if kept else text
    return result


def check_available_memory():
    memory_info = psutil.virtual_memory()
    available_memory = memory_info.available
    available_memory_gb = available_memory / (1024 ** 3)
    return available_memory_gb


#########################################################
# BATCHED VERSION: Dataset iterator with masking tokens #
# Added by A Howe
#########################################################

class BatchedTextSeqPairingDataset(Dataset):
    """
    Returns raw (text_caption, protein_sequence, accession_id).
    Tokenization is handled in collate_fn for batching.
    """

    def __init__(self, args, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

        self.text_captions = df["[final]text_caption"].tolist()
        self.protein_sequences = df[args.sequence_keyword].tolist()
        self.accession_ids = df[args.id_keyword].tolist()

        self.text_max_length = args.text_max_length
        self.seq_max_length = 1024

        # tokenizers (shared by collate_fn)
        self.text_tokenizer = AutoTokenizer.from_pretrained(
            args.text_model_path
        )
        _, self.sequence_tokenizer = pretrained.load_model_and_alphabet(
            args.seq_model_path
        )
        self.batch_converter = self.sequence_tokenizer.get_batch_converter()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return (
            self.text_captions[idx],
            self.protein_sequences[idx],
            self.accession_ids[idx],
        )

def collate_fn(
        batch, 
        dataset: BatchedTextSeqPairingDataset, 
        include_raw=False
):
    texts, sequences, accessions = zip(*batch)

    # -------- TEXT TOKENIZATION --------
    text_inputs = dataset.text_tokenizer.batch_encode_plus(
        list(texts),
        truncation=True,
        max_length=dataset.text_max_length,
        padding="max_length",
        return_tensors="pt",
        return_attention_mask=True,
        return_token_type_ids=False,
    )

    # -------- PROTEIN TOKENIZATION --------
    batch_sequences = list(zip(accessions, sequences))
    batch_labels, batch_strs, batch_tokens = dataset.batch_converter(batch_sequences)

    # truncate to max model length, but keep dynamic padding from batch_converter
    if batch_tokens.shape[1] > dataset.seq_max_length:
        batch_tokens = batch_tokens[:, : dataset.seq_max_length]

    if include_raw:
        return (
            text_inputs["input_ids"], 
            batch_tokens,
            list(texts),        # raw text captions
            list(sequences),    # raw protein sequences
            list(accessions),   # accession IDs
        )
    else:
        return text_inputs["input_ids"], batch_tokens

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


class MaskTextSeqPairing_Dataset(Dataset):

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
        self.mask_prob = 0.15  # probability of masking

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

        # apply masking
        text_inputs_masked = self.apply_masking_to_text(text_inputs['input_ids'])
        text_inputs['input_ids_masked'] = text_inputs_masked

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

        # apply masking
        protein_sequence_tokens_masked = self.apply_masking_to_protein(batch_tokens)

        sequence_inputs = {
            'protein_sequence_labels': batch_labels, # UniProtKB id
            'protein_sequence_str': batch_str, # original protein sequence (in amino acids)
            'protein_sequence_tokens': batch_tokens.long(), # training data
            'protein_sequence_tokens_masked': protein_sequence_tokens_masked.long() # training data for masks
        }

        return sequence_inputs

    def apply_masking_to_text(self, tokens: torch.Tensor) -> torch.Tensor:

        # mask some tokens and create the label sequence
        labels = []
        masked_tokens = tokens.clone()

        for i, token in enumerate(tokens.tolist()[0]):

            # skip masking if the tokens is a special token
            if token in [
                    self.text_tokenizer.cls_token_id,
                    self.text_tokenizer.sep_token_id,
                    self.text_tokenizer.pad_token_id,
                    self.text_tokenizer.mask_token_id]:
                continue

            # sample prob.
            prob = torch.rand(1).item()

            # mask token if prob is below mask_prob
            if prob < self.mask_prob:
                masked_tokens[0][i] = self.text_tokenizer.mask_token_id

            else:
                pass

        return masked_tokens


    def apply_masking_to_protein(self, tokens: torch.Tensor) -> torch.Tensor:
        # mask some tokens and create the label sequence
        labels = []
        masked_tokens = tokens.clone()

        # get the special token IDs
        cls_token_id = self.sequence_tokenizer.cls_idx
        sep_token_id = self.sequence_tokenizer.eos_idx
        pad_token_id = self.sequence_tokenizer.padding_idx
        unk_token_id = self.sequence_tokenizer.unk_idx
        mask_token_id = self.sequence_tokenizer.mask_idx

        for i, token in enumerate(tokens.tolist()[0]):

            # skip masking if the tokens is a special token
            if token in [
                    cls_token_id,
                    sep_token_id,
                    pad_token_id,
                    unk_token_id,
                    mask_token_id]:
                continue

            # sample prob.
            prob = torch.rand(1).item()

            # mask token if prob is below mask_prob
            if prob < self.mask_prob:

                masked_tokens[0][i] = self.sequence_tokenizer.mask_idx

            else:
                pass


        return masked_tokens


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
                protein_data['protein_sequence_tokens'],
                text_data['input_ids_masked'],
                protein_data['protein_sequence_tokens_masked']
        )


    def __len__(self):
        return self.length


#######################################################
# Dataset iterator with masking tokens + pfam dataset #
#######################################################


class Pfam_TextSeqPairing_Dataset(Dataset):

    def __init__(self, args: any, df: pd.Series, pfam_df: pd.Series):

        self.script_args = args
        # dataframe
        self.df = df
        self.length = self.df.shape[0]
        self.df_column_names = self.df.columns.tolist()
        self.protein_sequence_list = self.df[args.sequence_keyword].tolist()
        self.text_captions_list = self.df['[final]text_caption'].tolist()
        self.accession_id_list = self.df[args.id_keyword].tolist()
        self.pfam_labels_list = self.df['pfam_label'].tolist() # pfam labels found in the swiss-prot

        # Convert the strings into lists
        all_pf_codes = [ast.literal_eval(item) for item in self.pfam_labels_list]
        # Flatten the list of lists
        flat_pf_codes = [code for sublist in all_pf_codes for code in sublist]
        # Get unique PF codes in swiss-prot
        self.unique_pf_codes = list(set(flat_pf_codes))

        # protein family (pfam) database (over 40M sequences)
        self.grouped_pfam_df = pfam_df.groupby('pfam_label')

        pfam_group_keys = set(self.grouped_pfam_df.groups.keys())
        for label in self.unique_pf_codes:
            if label not in pfam_group_keys:
                assert f"Label {label} from swiss-prot is not in pfam_df!"
            else:
                pass

        # parameters
        self.text_max_length = args.text_max_length # max BERT sequence tokenization length
        self.seq_max_length = 1024 # max ESM model
        self.mask_prob = 0.15  # probability of masking

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

        # apply masking
        text_inputs_masked = self.apply_masking_to_text(text_inputs['input_ids'])
        text_inputs['input_ids_masked'] = text_inputs_masked

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

        # apply masking
        protein_sequence_tokens_masked = self.apply_masking_to_protein(batch_tokens)

        sequence_inputs = {
            'protein_sequence_labels': batch_labels, # UniProtKB id
            'protein_sequence_str': batch_str, # original protein sequence (in amino acids)
            'protein_sequence_tokens': batch_tokens.long(), # training data
            'protein_sequence_tokens_masked': protein_sequence_tokens_masked.long() # training data for masks
        }

        return sequence_inputs

    def apply_masking_to_text(self, tokens: torch.Tensor) -> torch.Tensor:

        # mask some tokens and create the label sequence
        labels = []
        masked_tokens = tokens.clone()

        for i, token in enumerate(tokens.tolist()[0]):

            # skip masking if the tokens is a special token
            if token in [
                    self.text_tokenizer.cls_token_id,
                    self.text_tokenizer.sep_token_id,
                    self.text_tokenizer.pad_token_id,
                    self.text_tokenizer.mask_token_id]:
                continue

            # sample prob.
            prob = torch.rand(1).item()

            # mask token if prob is below mask_prob
            if prob < self.mask_prob:
                masked_tokens[0][i] = self.text_tokenizer.mask_token_id

            else:
                pass

        return masked_tokens


    def apply_masking_to_protein(self, tokens: torch.Tensor) -> torch.Tensor:

        # mask some tokens and create the label sequence
        labels = []
        masked_tokens = tokens.clone()

        # get the special token IDs
        cls_token_id = self.sequence_tokenizer.cls_idx
        sep_token_id = self.sequence_tokenizer.eos_idx
        pad_token_id = self.sequence_tokenizer.padding_idx
        unk_token_id = self.sequence_tokenizer.unk_idx
        mask_token_id = self.sequence_tokenizer.mask_idx

        for i, token in enumerate(tokens.tolist()[0]):

            # skip masking if the tokens is a special token
            if token in [
                    cls_token_id,
                    sep_token_id,
                    pad_token_id,
                    unk_token_id,
                    mask_token_id]:
                continue

            # sample prob.
            prob = torch.rand(1).item()

            # mask token if prob is below mask_prob
            if prob < self.mask_prob:

                masked_tokens[0][i] = self.sequence_tokenizer.mask_idx

            else:
                pass

        return masked_tokens


    def extraction_pfam_samples(self, pfam_labels: list):
        """
        Extracts pfam samples from the provided labels.

        :param pfam_labels: A list containing pfam labels.
        :return: A tuple containing accession_id, Xp_pfam, Xt_pfam, and bool_pfam_vector.
        """

        # The function assumes that 'nan' is not present in pfam_labels
        bool_pfam_vector = ['True']

        queried_pfam_label = random.choice(pfam_labels)  # Directly get a random element
        temp_df = self.grouped_pfam_df.get_group(queried_pfam_label)

        sampled_row = temp_df.sample(n=1, random_state=self.script_args.seed).iloc[0]
        accession_id = str(sampled_row['id'])
        Xp_pfam = str(sampled_row['sequence'])
        Xt_pfam = str(sampled_row['[final]text_caption'])

        return (
            accession_id,
            Xp_pfam,
            Xt_pfam,
            bool_pfam_vector
        )

    def __getitem__(self, idx: torch.Tensor) -> (
            dict,
            dict
        ):

        # retrieve data samples
        protein_sequence = self.protein_sequence_list[idx] # protein sequences
        text_captions = self.text_captions_list[idx] # text captions
        accession_id = self.accession_id_list[idx] # sequence id from UniProt
        pfam_labels = ast.literal_eval(self.pfam_labels_list[idx]) # protein family labels


        #############################
        # Sample Swiss-prot dataset #
        #############################

        # prepare protein sequence in ESM format (e.g. tuple: (header, sequence)):
        batch_sequences = [
            (accession_id, protein_sequence)
        ]
        # stochastic annotation dropout (PenCL retraining)
        text_captions = apply_field_dropout(text_captions)

        # get swiss-prot protein-text pairing
        text_data = self.caption_tokenizer(batch_captions=[text_captions])
        protein_data = self.protein_tokenizer(batch_sequences=batch_sequences)

        #######################
        # Sample Pfam dataset #
        #######################

        if 'nan' in pfam_labels:
            pfam_accession_id = ''
            pfam_protein_sequence = ''
            pfam_text_captions = ''
            bool_pfam_vector = ['False']
            pfam_text_data = {'input_ids': []}
            pfam_protein_data = {'protein_sequence_tokens': []}
        else:
            pfam_accession_id, pfam_protein_sequence, pfam_text_captions, bool_pfam_vector = self.extraction_pfam_samples(pfam_labels=pfam_labels)

        # stochastic annotation dropout for Pfam text too
        pfam_text_captions = apply_field_dropout(pfam_text_captions)

        # get pfam protein-text pairing samples...
        pfam_batch_sequences = [(pfam_accession_id, pfam_protein_sequence)]
        pfam_text_data = self.caption_tokenizer(batch_captions=[pfam_text_captions])
        pfam_protein_data = self.protein_tokenizer(batch_sequences=pfam_batch_sequences)

        return (
                text_data['input_ids'],
                protein_data['protein_sequence_tokens'],
                text_data['input_ids_masked'],
                protein_data['protein_sequence_tokens_masked'],
                pfam_text_data['input_ids'],
                pfam_protein_data['protein_sequence_tokens'],
                pfam_text_data['input_ids_masked'],
                pfam_protein_data['protein_sequence_tokens_masked'],
                bool_pfam_vector
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
            logger.info("Number of GPUs: %s", self.trainer.world_size)
            logger.info("Current GPU index: %s", self.trainer.local_rank)

        # Load Swiss-Prot data
        df = self.load_swiss_prot()
        
        # Split the dataframe into train and valid sets
        train_df, valid_df = train_test_split(
            df,
            test_size=self.args.valid_size,
            random_state=self.args.seed
        )
 
        logger.info("Available memory after pfam_df: %s GB", check_available_memory())

        # Define datasets and dataloaders
        self.train_dataset = self.dataset_class(args=self.args, df=train_df)
        self.valid_dataset = self.dataset_class(args=self.args, df=valid_df)

    def load_swiss_prot(self) -> pd.Series:
        # Load and preprocess data (called on each GPU/TPU in DDP)
        logger.info('Load Swiss-Prot data...')

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


###################
# Pfam DataModule #
###################


class Pfam_DataModule(LightningDataModule):
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

    def _resolve_splits_dir(self) -> str:
        # Deviation from Rama's layout: prefer a user-specified splits dir so that
        # shard writes land inside the run directory instead of next to the input CSV.
        override = getattr(self.args, 'pfam_splits_dir', None)
        if override and str(override).lower() != 'none':
            return os.path.expanduser(str(override))
        directory_path = os.path.dirname(self.args.pfam_data_path)
        return f"{directory_path}/pfam_temp_splits"

    def prepare_data(self):
        import torch.distributed as dist

        # Use real distributed world_size (not PL's, which sees 1 in mpiexec mode)
        if dist.is_initialized():
            num_gpus = dist.get_world_size()
            my_rank = dist.get_rank()
        else:
            num_gpus = self.trainer.world_size if self.trainer else 1
            my_rank = 0

        splits_dir = self._resolve_splits_dir()

        # Only rank 0 does the splitting; other ranks wait
        if my_rank != 0:
            logger.info("Rank %s: waiting for rank 0 to prepare data splits...", my_rank)
            dist.barrier()
            return

        logger.info('Upload Pfam Database and split it over %s dataframes', num_gpus)
        # Load Swiss-Prot data
        df = self.load_swiss_prot()

        # Load Pfam data
        pfam_df = self.load_pfam_database()

        # Ensure Pfam labels match with Swiss-Prot
        pfam_unique_labels = set(pfam_df['pfam_label'].tolist())
        df = df[df['pfam_label'].apply(lambda x: all(label in pfam_unique_labels for label in ast.literal_eval(x)))]

        # Fast modular split: assign each row to a rank based on index
        # This is ~100x faster than stratified_split on 44M rows
        import numpy as np
        os.makedirs(splits_dir, exist_ok=True)
        assignments = np.arange(len(pfam_df)) % num_gpus
        for ii in range(num_gpus):
            split_df = pfam_df.iloc[assignments == ii]
            split_df.to_csv(f"{splits_dir}/split_pfam_rank_{ii}.csv", index=False)
            if ii == 0 or ii == num_gpus - 1:
                logger.info("  Split %s: %s rows", ii, len(split_df))

        logger.info("Saved %s Pfam splits to %s/", num_gpus, splits_dir)

        # After saving the splits to disk
        del pfam_df, df
        gc.collect()

        # Signal other ranks that splits are ready
        if dist.is_initialized():
            dist.barrier()

    def setup(self, stage=None):

        import torch.distributed as dist
        if dist.is_initialized():
            logger.info("Number of GPUs: %s", dist.get_world_size())
            logger.info("Current GPU index: %s", dist.get_rank())
        elif self.trainer is not None:
            logger.info("Number of GPUs: %s", self.trainer.world_size)
            logger.info("Current GPU index: %s", self.trainer.local_rank)

        # Load Swiss-Prot data
        df = self.load_swiss_prot()

        # Load Pfam data
        # Determine the GPU index — use real distributed rank for mpiexec mode
        if dist.is_initialized():
            gpu_idx = dist.get_rank()
        else:
            gpu_idx = self.trainer.local_rank if self.trainer else 0

        logger.info("Available memory before pfam_df: %s GB", check_available_memory())

        # Load the corresponding split pfam_df for this GPU
        splits_dir = self._resolve_splits_dir()
        columns_to_extract = ['id', 'pfam_label', 'sequence', '[final]text_caption']
        pfam_df = dd.read_csv(
                f"{splits_dir}/split_pfam_rank_{gpu_idx}.csv",
                dtype={6: 'str'},
                usecols=columns_to_extract
        ).compute()

        logger.info('Finished loading Pfam data...')
        pfam_df = pfam_df[pfam_df['sequence'].apply(lambda seq: len(seq) <= 1022)]

        # Ensure Pfam labels match with Swiss-Prot
        pfam_unique_labels = set(pfam_df['pfam_label'].tolist())
        df = df[df['pfam_label'].apply(lambda x: all(label in pfam_unique_labels for label in ast.literal_eval(x)))]

        # Split the dataframe into train and valid sets
        train_df, valid_df = train_test_split(
            df,
            test_size=self.args.valid_size,
            random_state=self.args.seed
        )

        logger.info("Available memory after pfam_df: %s GB", check_available_memory())

        # Define datasets and dataloaders
        self.train_dataset = self.dataset_class(args=self.args, df=train_df, pfam_df=pfam_df)
        self.valid_dataset = self.dataset_class(args=self.args, df=valid_df, pfam_df=pfam_df)

    def load_swiss_prot(self) -> pd.Series:
        # Load and preprocess data (called on each GPU/TPU in DDP)
        logger.info('Load Swiss-Prot data...')

        # Load Swiss-Prot data
        df = pd.read_csv(os.path.expanduser(self.args.data_path))
        df = df[df['protein_sequence'].apply(lambda seq: len(seq) <= 1022)]

        # remove OOD Test samples
        logger.info('Removing SwissProt OOD samples:')
        logger.info('SwissProt size: %s', df.shape[0])
        for ii, label in enumerate(self.OOD_pfam_labels):
            df = df[~df['pfam_label'].str.contains(label)]
            logger.info('SwissProt Size: %s', df.shape[0])

        logger.info('-' * 20)

        return df


    def load_pfam_database(self) -> pd.Series:

        logger.info('Load Pfam data...')
        # Step 1: Load Pfam dataset
        columns_to_extract = ['id', 'pfam_label', 'sequence', '[final]text_caption']
        pfam_df = dd.read_csv(self.args.pfam_data_path, dtype={6: 'str'}, usecols=columns_to_extract).compute()

        logger.info('Finished loading Pfam data with size %s...', pfam_df.shape[0])
        pfam_df = pfam_df[pfam_df['sequence'].apply(lambda seq: len(seq) <= 1022)]

        # remove OOD Test samples
        logger.info('Removing Pfam OOD samples:')
        logger.info('Pfam Size: %s', pfam_df.shape[0])
        for ii, label in enumerate(self.OOD_pfam_labels):
            pfam_df = pfam_df[~pfam_df['pfam_label'].str.contains(label)]
            logger.info('Pfam size: %s', pfam_df.shape[0])
        logger.info('-' * 20)

        return pfam_df


    def stratified_split(
            self,
            df: pd.Series,
            label_col: str,
            num_splits: int
            ) -> list:


        # count the number of instances for each class
        class_counts = df[label_col].value_counts()

        # Find classes that have only one instance
        small_classes = class_counts[class_counts < self.args.num_gpus].index

        # Duplicate the rows of singleton classes
        duplicated_rows = df[df['pfam_label'].isin(small_classes)]

        # Initialize an empty list to hold the smaller dfs
        smaller_dfs = []

        # Calculate the test size for each split
        test_size = 1.0 / num_splits

        # Duplicate rows for small classes so that each has at least num_gpus instances
        for small_class in tqdm(small_classes):

            small_class_df = df[df['pfam_label'] == small_class]
            num_duplications = self.args.num_gpus - small_class_df.shape[0]
            duplicated_small_class_df = pd.concat([small_class_df] * num_duplications, ignore_index=True)
            duplicated_rows = pd.concat([duplicated_rows, duplicated_small_class_df])


        # Append duplicated rows to the original DataFrame
        df = pd.concat([df, duplicated_rows], ignore_index=True)
        for ii in range(num_splits - 1):
            # Perform the stratified split
            train, test = train_test_split(df, stratify=df[label_col], test_size=test_size, random_state=self.args.seed)

            # Append the test (smaller set) to the list
            smaller_dfs.append(test)

            # Update df to be the remaining larger set for the next iteration
            df = train

            # Update test_size for the next iteration
            test_size = 1.0 / (num_splits - (ii + 1))

        # Append the last remaining set
        smaller_dfs.append(df)

        logger.info('Pfam database splits: %s', [len(temp_df) for temp_df in smaller_dfs])

        return smaller_dfs


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
            logger.info('Load both SwissProt and Pfam dataset...')
            self.train_dataset, self.valid_dataset, self.all_swiss_dataloader, self.all_pfam_dataloader = self.load_both()

        # get the swissprot dataset iterator
        elif args.pfam_data_path == 'None':
            logger.info('Load SwissProt dataset...')
            self.train_dataset, self.valid_dataset, self.all_swiss_dataloader = self.load_swissprot()
            self.all_pfam_dataloader = None

        # get the pfam dataset iterator 
        elif args.swissprot_data_path == 'None':
            logger.info('Load Pfam dataset...')
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


