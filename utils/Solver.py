import os, glob, inspect, time, math
import re
import zipfile
import random
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.utils import shuffle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import metrics


from numpy import savetxt

from torch.nn import functional as F
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.optim import Adam

data_path = os.path.join(os.getcwd(), 'resized_COVIDx')
data_frame_path = os.path.join(data_path, 'no_split.txt')
images_path = os.path.join(data_path, 'resized_COVIDx')

def make_dir(path):
    try: os.mkdir(path)
    except:
      pass

def prepare_directory(model_name):

    models_path = os.path.join(os.getcwd(),'Models')
    make_dir(models_path)
    dir_info = {}
    dir_info['model_name'] = model_name
    save_path =  os.path.join(models_path,model_name)
    make_dir(save_path)
    dir_info['save_path'] = save_path

    attr_list = ["params", "snapshots", "results"]
    for attr_name in attr_list:
        path = os.path.join(save_path, attr_name)
        make_dir(path)
        dir_info[attr_name] = path


    result_list = ["tr_restoring", "pca_latent_ontest", "tsne_latent_ontest", 'auc_graph']
    for result_name in result_list:
        path = os.path.join(dir_info['results'], result_name)
        make_dir(path)
        dir_info[result_name] = path

    onTrain_summary_path = os.path.join(save_path , "01-OnTrain-summary.csv")
    onTestsnapshots_summary_path = os.path.join(save_path , "02-OnTest-snapshots-summary.csv")


    if(not os.path.isfile(onTrain_summary_path)):
        with open(onTrain_summary_path, "w") as fcsv:
            fcsv.write("epoch,loss_MSE_G\n")

    if(not os.path.isfile(onTestsnapshots_summary_path)):
        with open(onTestsnapshots_summary_path, "w") as fcsv:
            fcsv.write("epoch,G_Normal_loss_Mean,G_AbNormal_loss_Mean,G_Normal_loss_SD,G_AbNormal_loss_SD,G_AUC\n")

    dir_info['onTrain_summary_path'] = onTrain_summary_path
    dir_info['onTestsnapshots_summary_path'] = onTestsnapshots_summary_path

    return dir_info


def unzipdata():
    covidx_zip = os.path.join(os.getcwd(),'resized_COVIDx.zip')
    covidx_folder = os.path.join(os.getcwd(),'resized_COVIDx')
    if(not os.path.isdir(covidx_folder)):
        make_dir(covidx_folder)
        zip_ref = zipfile.ZipFile(covidx_zip, 'r')
        zip_ref.extractall(covidx_folder)
        zip_ref.close()

def gray2rgb(gray):

    rgb = np.ones((gray.shape[0], gray.shape[1], 3)).astype(np.float32)
    rgb[:, :, 0] = gray[:, :, 0]
    rgb[:, :, 1] = gray[:, :, 0]
    rgb[:, :, 2] = gray[:, :, 0]

    return rgb

def dat2canvas(data):

    numd = math.ceil(np.sqrt(data.shape[0]))
    [dn, dh, dw, dc] = data.shape
    canvas = np.ones((dh*numd, dw*numd, dc)).astype(np.float32)

    for y in range(numd):
        for x in range(numd):
            try: tmp = data[x+(y*numd)]
            except: pass
            else: canvas[(y*dh):(y*dh)+224, (x*dw):(x*dw)+224, :] = tmp
    if(dc == 1):
        canvas = gray2rgb(gray=canvas)

    return canvas

def save_img(contents, names=["", "", ""], savename=""):

    num_cont = len(contents)
    plt.figure(figsize=(5*num_cont+2, 5))

    for i in range(num_cont):
        plt.subplot(1,num_cont,i+1)
        plt.title(names[i])
        plt.imshow(dat2canvas(data=contents[i]))

    plt.tight_layout()
    plt.savefig(savename)
    plt.close()

def discrete_cmap(N, base_cmap=None):

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)

    return base.from_list(cmap_name, color_list, N)


def boxplot(contents, savename=""):

    data, label = [], []
    for cidx, content in enumerate(contents):
        data.append(content)
        label.append("class-%d" %(cidx))

    plt.clf()
    fig, ax1 = plt.subplots()
    bp = ax1.boxplot(data, showfliers=True, whis=3)
    ax1.set_xticklabels(label, rotation=45)

    plt.tight_layout()
    plt.savefig(savename)
    plt.close()

def histogram(contents, savename=""):

    n1, _, _ = plt.hist(contents[0], bins=100, alpha=0.5, label='Normal')
    n2, _, _ = plt.hist(contents[1], bins=100, alpha=0.5, label='Abnormal')
    #h_inter = np.sum(np.minimum(n1, n2)) / np.sum(n1)
    plt.xlabel("MSE")
    plt.ylabel("Number of Data")
    xmax = max(contents[0].max(), contents[1].max())
    plt.xlim(0, xmax)
    plt.text(x=xmax * 0.01, y=max(n1.max(), n2.max()), s="Histogram")
    #plt.text(x=xmax*0.01, y=max(n1.max(), n2.max()), s="Histogram Intersection: %.3f" %(h_inter))
    plt.legend(loc='upper right')
    plt.savefig(savename)
    plt.close()

def save_graph(contents, xlabel, ylabel, savename):

    np.save(savename, np.asarray(contents))
    plt.clf()
    plt.rcParams['font.size'] = 15
    plt.plot(contents, color='blue', linestyle="-", label="loss")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout(pad=1, w_pad=1, h_pad=1)
    plt.savefig("%s.png" %(savename))
    plt.close()

def torch2npy(input):

    input = input.cpu()
    output = input.detach().numpy()
    return output

def loss_functions(x, x_hat):

    x, x_hat = x.cpu(), x_hat.cpu()
    criterion = torch.nn.MSELoss()
    loss = criterion(x_hat, x)

    return loss

def get_device():
  device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")


def train(G, dataset, epochs, batch_size, model_name, snapshot_number=10, lr=2e-4, load=False, snapshot=True, save_memory=True):
    optimizer_G = Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    dir_info = prepare_directory(model_name)
    # if a model is loaded, this number reflects the epochs of the previous training
    epoch_offset = 0
    if (load):
        pre_trained_models = os.listdir(dir_info['params'])

        param_G = os.path.join(dir_info['params'], pre_trained_models[-1])
        print('=======')
        print(param_G)
        print('=======')

        G.load_state_dict(torch.load(param_G))
        print("model loaded succesffully")
        epoch_offset = int(re.findall(r'\d+', pre_trained_models[-1])[0])
        print("---> Model was previously trained on {}-epochs".format(epoch_offset))

    print("\n<Training to %d new epochs (%d of minibatch size)>" % (epochs, batch_size))

    device = get_device()
    start_time = time.time()

    iteration = 0

    test_sq = 10
    test_size = test_sq ** 2
    list_recon = []
    restore_error = 0

    loss2npval = lambda loss: np.mean(loss.cpu().data.numpy()).round(4)
    loss_names = ["loss_MSE_G", ]

    G.train()

    for epoch in range(epoch_offset, epoch_offset + epochs):

        x_tr, x_tr_torch, y_tr, y_tr_torch, _ = dataset.next_train(batch_size=test_size)  # Initial batch

        x_restore = G(x_tr_torch)
        if(torch.cuda.is_available()):
            x_restore= x_restore.cuda()

        x_restore = np.transpose(torch2npy(x_restore), (0, 2, 3, 1))

        save_img(contents=[x_tr, x_restore, (x_tr - x_restore) ** 2], \
                 names=["Input\n(x)", "Restoration\n(x to x-hat)", "Difference"], \
                 savename=os.path.join(dir_info['results'], "tr_restoring", "%d.png" % (epoch)))

        while (True):
            if (iteration % 10 == 0):
                print("->Epoch {}, Training Index {}, Latest RestoreError {}".format(epoch, dataset.idx_train,
                                                                                     restore_error))

            x_tr, x_tr_torch, y_tr, y_tr_torch, terminator = dataset.next_train(batch_size)

            if (torch.cuda.is_available()):
                x_tr_torch = x_tr_torch.cuda()

            x_hat = G(x_tr_torch)

            G_loss = nn.MSELoss()(x_hat, x_tr_torch)
            restore_error = G_loss.item()

            optimizer_G.zero_grad()
            G_loss.backward()
            optimizer_G.step()

            list_recon.append(restore_error)

            x_hat = np.transpose(torch2npy(x_hat), (0, 2, 3, 1))

            iteration += 1
            if (terminator): break

        print("Epoch: {}, G_loss = {}".format(epoch, restore_error))

        with open(dir_info['onTrain_summary_path'], "a") as fcsv:
            fcsv.write("%d,%.6f\n" % (epoch, restore_error))

        if (epoch % snapshot_number == 0):
            if(save_memory):
                paramList = glob.glob(os.path.join(dir_info['params'], "*"))
                for param in paramList:
                    os.remove(param)

            torch.save(G.state_dict(), dir_info['params'] + "/params-epoch%d-G" % (epoch))

            if (snapshot):
                dataset.reset_idx()
                G.eval()
                snapshot_onTest(G, dataset, epoch, dir_info)
                if (torch.cuda.is_available()):
                    torch.cuda.empty_cache()
                G.train()

        elapsed_time = time.time() - start_time
        print("--->Elapsed: " + str(elapsed_time))




def latent_plot(latent, y, n, title, savename=""):
    plt.figure(figsize=(6, 5))
    plt.scatter(latent[:, 0], latent[:, 1], c=y, \
                marker='o', edgecolor='none', cmap=discrete_cmap(n, 'jet'))
    plt.colorbar(ticks=range(n))
    plt.grid()
    plt.tight_layout()
    plt.title(title)
    plt.savefig(savename)
    plt.close()


def snapshot_onTest(G, dataset, epoch, dir_info):
    G_scores_normal, G_scores_abnormal = [], []
    latent_vectors, labels = [], []
    G_scores = []

    snap_fcsv = open(dir_info['snapshots'] + "/01-OnTest-epoch{}-snapshot.csv".format(epoch), "w")

    snap_fcsv.write("class,G_score\n")

    print("Test SnapShot on epoch {}:".format(epoch))
    while (True):
        x, x_torch, y, y_torch, terminator = dataset.next_test(1)  # y_te does not used in this prj.

        x_hat = G(x_torch)
        if (torch.cuda.is_available()):
            x_hat = x_hat.cuda()
            x_torch = x_torch.cuda()

        x_enc = torch2npy(G.encode(x_torch))[0]
        x_G_score = loss_functions(x=x_torch, x_hat=x_hat).item()

        labels.append(y[0])
        latent_vectors.append(x_enc)
        G_scores.append(x_G_score)

        if (y[0] == 0):
            G_scores_normal.append(x_G_score)
        else:
            G_scores_abnormal.append(x_G_score)

        snap_fcsv.write("%d,%.7f\n" % (y, x_G_score))

        if (terminator): break

    # Saving the latent plots
    pca = PCA(n_components=2)
    tsne = TSNE(n_components=2)

    latent_vectors, labels = np.array(latent_vectors), np.array(labels)

    pca_features = pca.fit_transform(latent_vectors)
    tsne_features = tsne.fit_transform(latent_vectors)

    latent_plot(latent=pca_features, y=labels, n=dataset.num_class, title='PCA Plot-epoch%d' % (epoch), \
                savename=os.path.join(dir_info['pca_latent_ontest'], '%d.png' % (epoch)))

    latent_plot(latent=tsne_features, y=labels, n=dataset.num_class, title='t-SNE Plot-epoch%d' % (epoch), \
                savename=os.path.join(dir_info['tsne_latent_ontest'], '%d.png' % (epoch)))

    G_fpr, G_tpr, G_thresholds = metrics.roc_curve(labels, G_scores, pos_label=1)
    G_AUC = metrics.auc(G_fpr, G_tpr)
    plt.plot(G_fpr, G_tpr, label="AUC=" + str("%.3f" % G_AUC))
    plt.title('AUC for {}'.format(dir_info['model_name']))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc=4)
    plt.savefig(os.path.join(dir_info['auc_graph'], '%d.png' % (epoch)))
    plt.close()

    G_scores_normal = np.asarray(G_scores_normal)
    G_scores_abnormal = np.asarray(G_scores_abnormal)

    G_normal_avg, G_normal_std = np.average(G_scores_normal), np.std(G_scores_normal)
    G_abnormal_avg, G_abnormal_std = np.average(G_scores_abnormal), np.std(G_scores_abnormal)

    print("EPOCH: {}".format(epoch))

    print(" Generator Stats: ")
    print("   G-Noraml  avg: %.5f, std: %.5f" % (G_normal_avg, G_normal_std))
    print("   G-Abnoraml  avg: %.5f, std: %.5f" % (G_abnormal_avg, G_abnormal_std))
    print('   G-AUC for epoch{}: {:.5f}'.format(epoch, G_AUC))

    with open(dir_info['onTestsnapshots_summary_path'], "a") as fcsv:
        fcsv.write("%d,%.6f,%.6f,%.6f,%.6f,%.6f\n" \
                   % (epoch, G_normal_avg, G_abnormal_avg, G_normal_std, G_abnormal_std, G_AUC))




def test(G, dataset, model_name):
    dir_info = prepare_directory(model_name)

    pre_trained_models = os.listdir(dir_info['params'])

    param_G = os.path.join(dir_info['params'], pre_trained_models[-1])
    print('=======')
    print(param_G)
    print('=======')

    G.load_state_dict(torch.load(param_G))
    print("model loaded succesffully")
    epoch_offset = int(re.findall(r'\d+', pre_trained_models[-1])[0])
    print("---> Model was previously trained on {}-epochs".format(epoch_offset))

    dataset.reset_idx()
    G.eval()
    snapshot_onTest(G, dataset, epoch_offset, dir_info)
    if (torch.cuda.is_available()):
        torch.cuda.empty_cache()
    G.train()