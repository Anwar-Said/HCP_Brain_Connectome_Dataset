import nibabel as nib
import numpy as np
from nilearn.maskers import NiftiMasker,NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
from nilearn.datasets import fetch_atlas_schaefer_2018
from torch_geometric.data import Data, InMemoryDataset
import os,glob
# from nilearn.input_data import NiftiLabelsMasker
from nilearn.image import load_img
import torch
from nilearn import plotting
import networkx as nx
from scipy.sparse import coo_matrix
import pandas as pd
import zipfile
from torch_geometric.utils import degree
import io, shutil
from torch_geometric.data import Data
import os
import boto3
import pickle
## secret keys AWS s3 bucket
ACCESS_KEY = ''
SECRET_KEY = ''

# Set the HCP bucket name and file paths


# Create an S3 client
s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)

def crawl_read_data(source_dir, threshold):
    behavioral_df = pd.read_csv(os.path.join(source_dir,'HCP_behavioral.csv')).set_index('Subject')['Gender']
    behavioral_dict = behavioral_df.to_dict()
    dataset = []
    # files = glob.glob(os.path.join(source_dir,dirr,"*.zip"))
    target_path = "/home/anwar/disk/HCP_1200_crawled/"
    BUCKET_NAME = 'hcp-openaccess'
    FILE_PATH = 'HCP_1200/'
    with open("/home/anwar/disk/ids.pkl",'rb') as f:
        ids = pickle.load(f)
    roi = fetch_atlas_schaefer_2018(n_rois=1000)
    atlas = load_img(roi['maps'])
    print(type(atlas), atlas.shape)
    roi_masker = NiftiLabelsMasker(atlas)

    for iid in ids: 
        mri_file_path = "HCP_1200/"+iid+"/MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR.nii.gz"
        # s3.download_file(BUCKET_NAME, mri_file_path,os.path.join(target_path, iid+"_"+os.path.basename(mri_file_path)))
       
        image_path_LR = os.path.join(target_path, iid+"_"+os.path.basename(mri_file_path))
        img = load_img(image_path_LR)
        print(img.shape)
        
        timeseries= torch.tensor(roi_masker.fit_transform(img)).T
        
        attr = torch.sum(timeseries, dim=1)
        corref = corrcoef(timeseries)
        corref = corref.fill_diagonal_(0)

        label = behavioral_dict.get(int(iid))
        A = construct_Adj(corref, threshold)
        edge_index = A.nonzero().t().to(torch.long)
        mean, std = attr.mean(), attr.std()
        attribute = [(att-mean)/std for att in attr]
        attribute = torch.tensor(attribute, dtype= torch.float)
        
        y = 1 if label=="M" else 0
        data = Data(x = attribute, edge_index=edge_index, y = y)
        print(data.num_nodes,data.num_edges, data.num_node_features,data.has_isolated_nodes(), data.has_self_loops(),data.is_directed(),data.y,label)
        
        dataset.append(data)
    return dataset

def read_data(source_dir, threshold):
    behavioral_df = pd.read_csv(os.path.join(source_dir,'HCP_behavioral.csv')).set_index('Subject')['Gender']
    behavioral_dict = behavioral_df.to_dict()
    dataset = []
    # files = glob.glob(os.path.join(source_dir,dirr,"*.zip"))
    target_path = "/home/anwar/disk/HCP_1200_crawled/"
    BUCKET_NAME = 'hcp-openaccess'
    with open("/home/anwar/disk/ids.pkl",'rb') as f:
        ids = pickle.load(f)
    roi = fetch_atlas_schaefer_2018(n_rois=1000)
    atlas = load_img(roi['maps'])
    roi_masker = NiftiLabelsMasker(atlas)
    count = 0
    for iid in ids: 
        try:
            mri_file_path = "HCP_1200/"+iid+"/MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR.nii.gz"
            if not os.path.exists(os.path.join(target_path, iid+"_"+os.path.basename(mri_file_path))):
                s3.download_file(BUCKET_NAME, mri_file_path,os.path.join(target_path, iid+"_"+os.path.basename(mri_file_path)))
            image_path_LR = os.path.join(target_path, iid+"_"+os.path.basename(mri_file_path))
            img = load_img(image_path_LR)
            # print(img.shape)
            timeseries= torch.tensor(roi_masker.fit_transform(img)).T
            
            # attr = torch.sum(timeseries, dim=1)
            corref = torch.corrcoef(timeseries)
            corref = corref.fill_diagonal_(0).to(torch.float)

            label = behavioral_dict.get(int(iid))
            A = construct_Adj_postive_perc(corref, threshold)
            edge_index = A.nonzero().t().to(torch.long)
            
            # mean, std = attr.mean(), attr.std()
            # attribute = [(att-mean)/std for att in attr]
            # attribute = torch.tensor(attribute, dtype= torch.float)
            
            y = 1 if label=="M" else 0
            data = Data(x = corref, edge_index=edge_index, y = y)
            print(data.num_nodes,data.num_edges, data.num_node_features,data.has_isolated_nodes(), data.has_self_loops(),data.is_directed(),data.y,label)
            
            dataset.append(data)
            count +=1
        except:
            print("skipped")
    return dataset


    

def corrcoef(x):
    mean_x = torch.mean(x, 1, keepdim=True)
    xm = x.sub(mean_x.expand_as(x))
    c = xm.mm(xm.t())
    c = c / (x.size(1) - 1)
    d = torch.diag(c)
    stddev = torch.pow(d, 0.5)
    c = c.div(stddev.expand_as(c))
    c = c.div(stddev.expand_as(c).t())
    c = torch.clamp(c, -1.0, 1.0)
    return c

def construct_Adj(corref,threshold):
    flattened = corref.flatten()
    threshold = torch.quantile(flattened, threshold)
    A = (corref >=threshold).float()
    A = torch.mul(A, corref)
    A[A>0] = 1
    # graph = nx.from_scipy_sparse_matrix(coo_matrix(A))
    return A
def construct_Adj_postive_perc(corr, perc):
    corr_matrix_copy = corr.detach().clone()
    threshold = np.percentile(corr_matrix_copy[corr_matrix_copy > 0], 100 - perc)
    corr_matrix_copy[corr_matrix_copy < threshold] = 0
    corr_matrix_copy[corr_matrix_copy >= threshold] = 1
    return corr_matrix_copy

def construct_Adj_threshold(corref, threshold):
    A = corref.detach().clone()
    A[A>=threshold] = 1
    A[A<threshold] = 0
    return A
class Brain_Connectome(InMemoryDataset):
    def __init__(self, root,source_dir,dirr, threshold,transform=None, pre_transform=None, pre_filter=None):
        self.source_dir,self.dirr, self.threshold = source_dir,dirr, threshold
        super().__init__(root, transform, pre_transform, pre_filter)
        
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        # return ['rest1_processed_gender.pt']
        return ['gender_brain_dataset.pt']

#     def download(self):
#         # Download to `self.raw_dir`.
#         download_url(url, self.raw_dir)
#         ...

    def process(self):
        behavioral_df = pd.read_csv(os.path.join(self.source_dir,'HCP_behavioral.csv')).set_index('Subject')['Gender']
        behavioral_dict = behavioral_df.to_dict()
        data_list = []
        files = glob.glob(os.path.join(self.source_dir,self.dirr,"*.zip"))
        target_path = os.getcwd()
        roi = fetch_atlas_schaefer_2018(data_dir=os.path.join(self.source_dir, 'roi'))
        roi_masker = NiftiLabelsMasker(load_img(roi['maps']))
        count = 0
        for f in files:
            with zipfile.ZipFile(f,'r') as zip_file:
                zip_file.extractall(target_path)
            id = f.split(".")[0].split("/")[-1].split("_")[0]
            subdirectory_path_LR = os.path.join(target_path, id+"/MNINonLinear/Results/rfMRI_REST1_LR")
            name_LR = 'rfMRI_REST1_LR.nii.gz'
            image_path_LR = os.path.join(subdirectory_path_LR,name_LR)
            try:
                timeseries_LR= torch.tensor(roi_masker.fit_transform(load_img(image_path_LR)))
                attr_LR = torch.sum(timeseries_LR, dim=1)
                corref_LR = corrcoef(timeseries_LR)
                corref_LR = corref_LR.fill_diagonal_(0)
                A_LR = construct_Adj(corref_LR, self.threshold)
                edge_index_LR = A_LR.nonzero().t().to(torch.long)
                mean_LR, std_LR = attr_LR.mean(), attr_LR.std()
                attribute_LR = [(att-mean_LR)/std_LR for att in attr_LR]
                attribute_LR = torch.tensor(attribute_LR, dtype= torch.float)
                label = behavioral_dict.get(int(id))
                y = 1 if label=="M" else 0
                dataLR = Data(x = attribute_LR, edge_index=edge_index_LR, y = y)
                data_list.append(dataLR)
            except:
                print("{} not found".format(image_path_LR))
            subdirectory_path_RL = os.path.join(target_path, id+"/MNINonLinear/Results/rfMRI_REST1_RL")
            name_RL = 'rfMRI_REST1_RL.nii.gz'
            image_path_RL = os.path.join(subdirectory_path_RL,name_RL)
            try:
                img = load_img(image_path_RL)
                timeseries_RL= torch.tensor(roi_masker.fit_transform(img))
                attr_RL = torch.sum(timeseries_RL, dim=1)
                corref_RL = corrcoef(timeseries_RL)
                corref_RL = corref_RL.fill_diagonal_(0)
                label = behavioral_dict.get(int(id))
                A_RL = construct_Adj(corref_RL, self.threshold)
                edge_index_RL = A_RL.nonzero().t().to(torch.long)
                # print("edgeindex shape: ",edge_index_LR.shape)
                mean_RL, std_RL = attr_RL.mean(), attr_RL.std()
                attribute_RL = [(att-mean_RL)/std_RL for att in attr_RL]
                attribute_RL = torch.tensor(attribute_RL, dtype= torch.float)
                y = 1 if label=="M" else 0
                dataRL = Data(x = attribute_RL, edge_index=edge_index_RL, y = y)
                data_list.append(dataRL)
                shutil.rmtree(os.path.join(target_path,id))
            except:
                print("{} not found".format(image_path_RL))    
            count +=1
            print("{} samples processed!".format(count), len(data_list))
        print("length of datalist:", len(data_list))
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        print("saving path:",self.processed_paths[0])
        torch.save((data, slices), self.processed_paths[0])

class Brain_Connectome_Dataset(InMemoryDataset):
    def __init__(self, root,dataset_name,source_dir, threshold,transform=None, pre_transform=None, pre_filter=None):
        self.source_dir, self.dataset_name,self.threshold = source_dir, dataset_name,threshold
        super().__init__(root, transform, pre_transform, pre_filter)
        
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        # return ['rest1_processed_gender.pt']
        return [self.dataset_name+'.pt']

#     def download(self):
#         # Download to `self.raw_dir`.
#         download_url(url, self.raw_dir)
#         ...

    def process(self):
        behavioral_df = pd.read_csv(os.path.join(self.source_dir,'HCP_behavioral.csv')).set_index('Subject')['Gender']
        behavioral_dict = behavioral_df.to_dict()
        dataset = []
        # files = glob.glob(os.path.join(source_dir,dirr,"*.zip"))
        target_path = "/home/anwar/disk/HCP_1200_crawled/"
        BUCKET_NAME = 'hcp-openaccess'
        with open("/home/anwar/disk/ids.pkl",'rb') as f:
            ids = pickle.load(f)
        roi = fetch_atlas_schaefer_2018(n_rois=1000)
        atlas = load_img(roi['maps'])
        roi_masker = NiftiLabelsMasker(atlas)
        count = 0
        for iid in ids: 
            try:
                mri_file_path = "HCP_1200/"+iid+"/MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR.nii.gz"
                if not os.path.exists(os.path.join(target_path, iid+"_"+os.path.basename(mri_file_path))):
                    s3.download_file(BUCKET_NAME, mri_file_path,os.path.join(target_path, iid+"_"+os.path.basename(mri_file_path)))
                image_path_LR = os.path.join(target_path, iid+"_"+os.path.basename(mri_file_path))
                img = load_img(image_path_LR)
                # print(img.shape)
                timeseries= torch.tensor(roi_masker.fit_transform(img)).T
                attr = torch.sum(timeseries, dim=1)
                corref = torch.corrcoef(timeseries)
                corref = corref.fill_diagonal_(0).to(torch.float)
                label = behavioral_dict.get(int(iid))
                A = construct_Adj_postive_perc(corref, self.threshold)
                edge_index = A.nonzero().t().to(torch.long)
                
                mean, std = attr.mean(), attr.std()
                attribute = [(att-mean)/std for att in attr]
                attribute = torch.tensor(attribute, dtype= torch.float)
                
                y = 1 if label=="M" else 0
                data = Data(x = attribute, edge_index=edge_index, y = y)
                print(data.num_nodes,data.num_edges, data.num_node_features,data.has_isolated_nodes(), data.has_self_loops(),data.is_directed(),data.y,label)
                
                dataset.append(data)
                count +=1
                # if count==2:
                #     break
            except:
                print("individual {} skipped!".format(iid))
            print("{} processed!".format(count) )
        print(len(dataset))
        if self.pre_filter is not None:
            dataset = [data for data in dataset if self.pre_filter(data)]

        if self.pre_transform is not None:
            dataset = [self.pre_transform(data) for data in dataset]

        data, slices = self.collate(dataset)
        print("saving path:",self.processed_paths[0])
        torch.save((data, slices), self.processed_paths[0])

class Brain_Connectome_Task_Dataset(InMemoryDataset):
    def __init__(self, root,dataset_name,source_dir, threshold,transform=None, pre_transform=None, pre_filter=None):
        self.source_dir, self.dataset_name,self.threshold = source_dir, dataset_name,threshold
        super().__init__(root, transform, pre_transform, pre_filter)
        
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        # return ['rest1_processed_gender.pt']
        return [self.dataset_name+'.pt']

#     def download(self):
#         # Download to `self.raw_dir`.
#         download_url(url, self.raw_dir)
#         ...

    def process(self):
        # behavioral_df = pd.read_csv(os.path.join(self.source_dir,'HCP_behavioral.csv')).set_index('Subject')['Gender']
        # behavioral_dict = behavioral_df.to_dict()
        dataset = []
        # files = glob.glob(os.path.join(source_dir,dirr,"*.zip"))
        target_path = "/home/anwar/disk/HCP_crawled_Task/"
        BUCKET_NAME = 'hcp-openaccess'
        with open("/home/anwar/disk/ids.pkl",'rb') as f:
            ids = pickle.load(f)
        roi = fetch_atlas_schaefer_2018(n_rois=1000)
        atlas = load_img(roi['maps'])
        roi_masker = NiftiLabelsMasker(atlas)
        count = 0
        for iid in ids: 
            
            rest_file_path = "HCP_1200/"+iid+"/MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR.nii.gz"
            emotion_path = "HCP_1200/"+iid+"/MNINonLinear/Results/tfMRI_EMOTION_LR/tfMRI_EMOTION_LR.nii.gz"
            gambling_path = "HCP_1200/"+iid+"/MNINonLinear/Results/tfMRI_GAMBLING_LR/tfMRI_GAMBLING_LR.nii.gz"
            language_path = "HCP_1200/"+iid+"/MNINonLinear/Results/tfMRI_LANGUAGE_LR/tfMRI_LANGUAGE_LR.nii.gz"
            motor_path = "HCP_1200/"+iid+"/MNINonLinear/Results/tfMRI_MOTOR_LR/tfMRI_MOTOR_LR.nii.gz"
            relational_path = "HCP_1200/"+iid+"/MNINonLinear/Results/tfMRI_RELATIONAL_LR/tfMRI_RELATIONAL_LR.nii.gz"
            social_path = "HCP_1200/"+iid+"/MNINonLinear/Results/tfMRI_SOCIAL_LR/tfMRI_SOCIAL_LR.nii.gz"
            wm_path = "HCP_1200/"+iid+"/MNINonLinear/Results/tfMRI_WM_LR/tfMRI_WM_LR.nii.gz"
            all_paths = [emotion_path,gambling_path,language_path,motor_path,relational_path,social_path,wm_path]
            for y, path in enumerate(all_paths):
                try:
                    if not os.path.exists(os.path.join(target_path, iid+"_"+os.path.basename(path))):
                        
                        s3.download_file(BUCKET_NAME, path,os.path.join(target_path, iid+"_"+os.path.basename(path)))
                    image_path_LR = os.path.join(target_path, iid+"_"+os.path.basename(path))
                    img = load_img(image_path_LR)
                    # print(img.shape)
                    timeseries= torch.tensor(roi_masker.fit_transform(img)).T
                    attr = torch.sum(timeseries, dim=1)
                    corref = torch.corrcoef(timeseries)
                    corref = corref.fill_diagonal_(0).to(torch.float)
                    label = y
                    A = construct_Adj_postive_perc(corref, self.threshold)
                    edge_index = A.nonzero().t().to(torch.long)
                    
                    mean, std = attr.mean(), attr.std()
                    attribute = [(att-mean)/std for att in attr]
                    attribute = torch.tensor(attribute, dtype= torch.float)
                    
                    # y = 1 if label=="M" else 0
                    data = Data(x = attribute, edge_index=edge_index, y = y)
                    # print(data.num_nodes,data.num_edges, data.num_node_features,data.has_isolated_nodes(), data.has_self_loops(),data.is_directed(),data.y,label)
                    dataset.append(data)
                    os.remove(image_path_LR)
                except:
                    print("instance not found")
            count +=1
                    # if count==2:
                    #     break
            print("{} individuals processed!".format(count),len(dataset))
        print(len(dataset))
        if self.pre_filter is not None:
            dataset = [data for data in dataset if self.pre_filter(data)]

        if self.pre_transform is not None:
            dataset = [self.pre_transform(data) for data in dataset]

        data, slices = self.collate(dataset)
        print("saving path:",self.processed_paths[0])
        torch.save((data, slices), self.processed_paths[0])

