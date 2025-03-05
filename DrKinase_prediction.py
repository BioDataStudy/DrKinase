import os, re
import argparse
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from keras.models import model_from_json
from math import log
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import percentileofscore
from pathlib import Path
import multiprocessing.pool
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

both_embeddings = True

multiallelic_dict = {
        "Gatekeeper": "Gatekeeper",
        "A-loop": "A_loop",
        "G-loop": "G_loop",
        "aC-Helix": "aC_Helix"}

multiallelic_peptide_lengths = {
        "A-loop": [11],
        "αC-Helix": [6]}

def read_fasta(fasta_file):
    try:
        fp = open(fasta_file)
    except IOError:
        exit()
    else:
        fp = open(fasta_file)
        lines = fp.readlines()
        fasta_dict = {} 
        gene_id = ""
        for line in lines:
            if line[0] == '>':
                if gene_id != "":
                    fasta_dict[gene_id] = seq
                seq = ""
                gene_id = line.strip()[1:]
            else:
                seq += line.strip()        
        fasta_dict[gene_id] = seq       
    return fasta_dict  

def sample_fasta_peptides(sequences, types):

    data_dict = defaultdict(defaultdict)
    
    for (i, (name, sequence)) in enumerate(sequences.items()):
        if not isinstance(sequence, str):
            raise ValueError("Expected string, not %s (%s)" % (
                sequence, type(sequence)))
        for t in types:
            if t in ['Gatekeeper', 'G-loop']:
                continue
            peptide_lengths = multiallelic_peptide_lengths[t]
            
            for peptide_start in range(len(sequence) - min(peptide_lengths) + 1):
                for peptide_length in peptide_lengths:
                    peptide = sequence[peptide_start: peptide_start + peptide_length]
                    peptide_start_stop = '_'.join([peptide, str(peptide_start + 1), str(peptide_start + peptide_length)])
                    if len(peptide) != peptide_length:
                        continue

                    if name not in data_dict.keys() or t not in data_dict[name]:
                        data_dict[name][t] = [peptide_start_stop]
                    else:
                        data_dict[name][t].append(peptide_start_stop)

    return data_dict 

def AA_encoding(seq_extended):
    amino = "ABCDEFGHIJKLMNOPQRSTUVWXYZ-"
    encoder = LabelEncoder()
    encoder.fit(list(amino))
    seq_transformed = np.array(
        list(map(encoder.transform, np.array([list(i.upper()) for i in seq_extended]))))   
    return seq_transformed[0]

def import_model_comprehensive(main_dir = 'models/models_comprehensive'):
    models = {}
    pred_type =['Gatekeeper', 'A-loop', 'G-loop', 'αC-Helix']
    for t in pred_type:
        json_f = open(main_dir + "/%s_DL.json" % t, 'r')
        loaded_model_json = json_f.read()
        json_f.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(main_dir + '/model_%s.h5' % t)
        models[t] = loaded_model
    return models

def import_model_simple(main_dir = 'models/models_light'):
    models = {}
    pred_type =['Gatekeeper', 'A-loop', 'G-loop', 'αC-Helix']
    for t in pred_type:
        json_f = open(main_dir + "/%s_DL.json" % t, 'r')
        loaded_model_json = json_f.read()
        json_f.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(main_dir + '/model_%s.h5' % t)
        models[t] = loaded_model
    return models

def split_embedding_seq(embeddings, slicesize, shift):
    splited_em_seq = []
    for i in range(0, len(embeddings) - slicesize + 1):
        splited_em_seq.append([i + (slicesize // 2) - shift, embeddings[i:i + slicesize]])
        
    return np.array(splited_em_seq)

target_freq = {
    'M': 0.36416184971098264,
    'T': 0.21098265895953758,
    'L': 0.17341040462427745,
    'F': 0.14450867052023122,
    'V': 0.03468208092485549
}

target_freq1 = ['I', 'S', 'Y', 'G', 'N', 'W', 'Q', 'C', 'R', 'K']
target_freq2 = ['A', 'D', 'E', 'H', 'P']

suppress_factor1 = 0.7
suppress_factor2 = 0.1

def adjust_scores(predictions, residues):
    adjusted = []
    for score, residue in zip(predictions, residues):
        if residue in target_freq:
            if score > 0.49:
                freq = target_freq[residue]
                new_score = score + (1 - score) * freq
            else:
                new_score = score
        elif residue in target_freq1:
            new_score = score * suppress_factor1
        else:
            new_score = score * suppress_factor2
        
        adjusted.append(new_score)
    return adjusted

    
####################################################################################################################
#################################################  select model  ###################################################
####################################################################################################################

def pred_and_write_metrics_datatable_sel_mod(fasta_data, outdir, pred_type, both_embeddings="no"):

    # ID_sequences = fasta_data
    ID_sequences = read_fasta(fasta_data)
    ID_type_peptides_position = sample_fasta_peptides(ID_sequences, pred_type)
    ps_results = []

    if both_embeddings == "yes":  

        from allennlp.commands.elmo import ElmoEmbedder
        import torch

        class Elmo_embedder():
            def __init__(self, model_dir="model/uniref50_v2", weights="weights.hdf5",
                        options="options.json", threads=1000):
                if threads == 1000:
                    torch.set_num_threads(multiprocessing.cpu_count() // 2)
                else:
                    torch.set_num_threads(threads)

                self.model_dir = Path(model_dir)
                self.weights = self.model_dir / weights
                self.options = self.model_dir / options
                self.seqvec = ElmoEmbedder(self.options, self.weights, cuda_device=-1)

            def elmo_embedding(self, x, start=None, stop=None):
                assert start is None and stop is None, "deprecated to use start stop, please trim seqs beforehand"

                if type(x[0]) == str:
                    x = np.array([list(i.upper()) for i in x])
                embedding = self.seqvec.embed_sentences(x)
                X_parsed = []
                for i in embedding:
                    X_parsed.append(i.mean(axis=0))
                return X_parsed
        
        models = import_model_comprehensive()
        elmo_embedder = Elmo_embedder(threads=60)
        
        for uid in ID_sequences.keys():
            print("===========================")
            print(f'Running... {uid}')
            
            outdir_pred = outdir +"/" + uid + '/'
            if not os.path.isdir(outdir_pred):
                os.makedirs(outdir_pred) 

            for t in pred_type:
                
                slicesize = 21 if t == 'Gatekeeper' else 25
                shift = 10 if t == 'Gatekeeper' else 12
                
                seq_local = ID_sequences[uid]
                seq_len = len(seq_local)
                seq_local = seq_local.upper()
                
                seq_local_list = np.array(list(seq_local))
                X_embedding = elmo_embedder.elmo_embedding(seq_local_list)
                protein_pad_global = np.zeros((seq_len + (shift * 2), 1024), dtype=np.float32)

                for i in range(0, seq_len, 1):
                    protein_pad_global[i + (shift)] = X_embedding[i]

                protein_pad_local = ["-"] * (seq_len + (shift * 2))
                for i in range(0, seq_len, 1):
                    protein_pad_local[i + (shift)] = seq_local[i]
                    
                protein_pad_local = "".join(protein_pad_local)        
                
                if t == 'Gatekeeper':
                    
                    seq_slices = split_embedding_seq(np.array(protein_pad_global), slicesize, shift)
                    positions = seq_slices[:, 0]
                    X_test = np.stack(seq_slices[:, 1])
                    
                    if both_embeddings:
                                        
                        splited_AA_seqs = []
                        for i in range(0, len(protein_pad_local) - slicesize + 1):
                            splited_AA_seqs.append(protein_pad_local[i:i + slicesize])

                        seq_transformed = np.array(list(map(lambda x: AA_encoding([x]), splited_AA_seqs)))
                        
                        X_test1 = [X_test, seq_transformed]
                    
                    probs_ = models[t].predict(X_test1) 
                    final_probs_  = adjust_scores(probs_, seq_local)

                    out = f'{outdir_pred}/{t}.txt'
                    with open(out, 'w') as outfile1:
                        outfile1.write('Position\tAmino acid\tScore\n')
                        for x in range(len(final_probs_)):
                            outfile1.write(f'{x+1}\t{seq_local[x]}\t{final_probs_[x][0]}\n')
                    
                    Gatekeeper_r_p = pd.read_csv(out, sep="\t")
                    
                    Gatekeeper_r_p['Entry'] = uid
                    Gatekeeper_r_p['Start'] = Gatekeeper_r_p['Position']
                    Gatekeeper_r_p['Seq'] = splited_AA_seqs

                    Gatekeeper_r_p = Gatekeeper_r_p.rename(columns = {'Position':'End','Amino acid':'Hit'})
                    Gatekeeper_r_p = Gatekeeper_r_p[["Entry","Hit","Seq","Start","End","Score"]]
                    Gatekeeper_r_p['Type'] = t
                    ps_results.append(Gatekeeper_r_p)

                    print('%s prediction done...' % t)
                    
                else:
                    if t == 'G-loop':
                        pattens = 'G.G..G'
                        G_results = []
                        for m in re.finditer(pattens, seq_local):
                            G_results.append([uid, m.group(), m.start()+1, m.end()])
                        
                        if len(G_results) > 0:
                            g_loop_r, all_seq_transformed, all_seq_elmo_embedding = [], [], []
                            for hit in G_results:
                                start_origin = hit[2]-1
                                stop_origin = hit[3]
                                start = start_origin + shift
                                stop = stop_origin + shift
                                median_pos = (start+stop-1)//2
                                slice_start = median_pos - slicesize // 2
                                slice_stop = slice_start + slicesize
                                query_seq = protein_pad_local[slice_start:slice_stop]
                                seq_transformed = AA_encoding([query_seq])
                                all_seq_transformed.append(seq_transformed)
                                seq_elmo_embedding = protein_pad_global[slice_start:slice_stop]
                                all_seq_elmo_embedding.append(seq_elmo_embedding)
                                g_loop_r.append([uid, hit[1], query_seq, start_origin+1, stop_origin])

                            probs_ = models[t].predict([all_seq_elmo_embedding, all_seq_transformed]) 

                            g_loop_r_p = pd.DataFrame(g_loop_r,columns=["Entry","Hit","Seq","Start","End"])             
                            g_loop_r_p['Score'] = probs_
                            g_loop_r_p = g_loop_r_p.sort_values(by='Score', ascending=False)
                            path_output_G_loop = outdir_pred + '/%s.txt' % t
                            g_loop_r_p.to_csv(path_output_G_loop, sep="\t",index=False)
                        
                            g_loop_r_p['Type'] = t
                            ps_results.append(g_loop_r_p)
                        else:
                            g_loop_r = []
                            g_loop_r_p = pd.DataFrame(g_loop_r,columns=["Entry","Hit","Seq","Start","End","Score"]) 
                            path_output_G_loop = outdir_pred + '/%s.txt' % t
                            g_loop_r_p.to_csv(path_output_G_loop, sep="\t",index=False)

                        print('%s prediction done...' % t)

                    else:
                        
                        AH_loop_r = [] 
                        all_seq_transformed, all_seq_elmo_embedding = [], []
                        AH_pred_infor = ID_type_peptides_position[uid][t]
                        
                        for g in AH_pred_infor:
                            hit = g.split('_')
                            start_origin = int(hit[1])-1
                            stop_origin = int(hit[2])
                            start = start_origin + shift
                            stop = stop_origin + shift
                            median_pos = (start+stop-1)//2
                            slice_start = median_pos - slicesize // 2
                            slice_stop = slice_start + slicesize
                            query_seq = protein_pad_local[slice_start:slice_stop]
                            seq_transformed = AA_encoding([query_seq])
                            all_seq_transformed.append(seq_transformed)
                            seq_elmo_embedding = protein_pad_global[slice_start:slice_stop]
                            all_seq_elmo_embedding.append(seq_elmo_embedding)
                            AH_loop_r.append([uid, hit[0], query_seq, start_origin+1, stop_origin])

                        probs_ = models[t].predict([all_seq_elmo_embedding, all_seq_transformed]) 
                        
                        AH_loop_r_p = pd.DataFrame(AH_loop_r,columns=["Entry","Hit","Seq","Start","End"])             
                        AH_loop_r_p['Score'] = probs_
                        AH_loop_r_p = AH_loop_r_p.sort_values(by='Score', ascending=False)
                        path_output_AH_loop = outdir_pred + '/%s.txt' % t
                        AH_loop_r_p.to_csv(path_output_AH_loop, sep="\t",index=False)
                        
                        AH_loop_r_p['Type'] = t                
                        ps_results.append(AH_loop_r_p)
                        
                        print('%s prediction done...' % t)             


    # fast version
    else:
        models = import_model_simple()

        for uid in ID_sequences.keys():
            print("===========================")
            print(f'Running... {uid}')
            
            outdir_pred = outdir +"/" + uid + '/'
            if not os.path.isdir(outdir_pred):
                os.makedirs(outdir_pred) 

            for t in pred_type:
                
                slicesize = 21 if t == 'Gatekeeper' else 25
                shift = 10 if t == 'Gatekeeper' else 12
                
                seq_local = ID_sequences[uid]
                seq_len = len(seq_local)
                seq_local = seq_local.upper()

                protein_pad_local = ["-"] * (seq_len + (shift * 2))
                for i in range(0, seq_len, 1):
                    protein_pad_local[i + (shift)] = seq_local[i]
                    
                protein_pad_local = "".join(protein_pad_local)        
                
                if t == 'Gatekeeper':
                                        
                    splited_AA_seqs = []
                    for i in range(0, len(protein_pad_local) - slicesize + 1):
                        splited_AA_seqs.append(protein_pad_local[i:i + slicesize])

                    seq_transformed = np.array(list(map(lambda x: AA_encoding([x]), splited_AA_seqs)))
                                        
                    probs_ = models[t].predict(seq_transformed) 

                    final_probs_  = adjust_scores(probs_, seq_local)

                    out = f'{outdir_pred}/{t}.txt'
                    with open(out, 'w') as outfile1:
                        outfile1.write('Position\tAmino acid\tScore\n')
                        for x in range(len(final_probs_)):
                            outfile1.write(f'{x+1}\t{seq_local[x]}\t{final_probs_[x][0]}\n')
                    
                    Gatekeeper_r_p = pd.read_csv(out, sep="\t")
                    
                    Gatekeeper_r_p['Entry'] = uid
                    Gatekeeper_r_p['Start'] = Gatekeeper_r_p['Position']
                    Gatekeeper_r_p['Seq'] = splited_AA_seqs

                    Gatekeeper_r_p = Gatekeeper_r_p.rename(columns = {'Position':'End','Amino acid':'Hit'})
                    Gatekeeper_r_p = Gatekeeper_r_p[["Entry","Hit","Seq","Start","End","Score"]]
                    Gatekeeper_r_p['Type'] = t
                    ps_results.append(Gatekeeper_r_p)

                    print('%s prediction done...' % t)
                    
                else:
                    if t == 'G-loop':
                        pattens = 'G.G..G'
                        G_results = []
                        for m in re.finditer(pattens, seq_local):
                            G_results.append([uid, m.group(), m.start()+1, m.end()])
                        
                        if len(G_results) > 0:
                            # g_loop_r, all_seq_transformed, all_seq_elmo_embedding = [], [], []
                            g_loop_r, all_seq_transformed = [], []
                            for hit in G_results:
                                start_origin = hit[2]-1
                                stop_origin = hit[3]
                                start = start_origin + shift
                                stop = stop_origin + shift
                                median_pos = (start+stop-1)//2
                                slice_start = median_pos - slicesize // 2
                                slice_stop = slice_start + slicesize
                                query_seq = protein_pad_local[slice_start:slice_stop]
                                seq_transformed = AA_encoding([query_seq])
                                all_seq_transformed.append(seq_transformed)
                                # seq_elmo_embedding = protein_pad_global[slice_start:slice_stop]
                                # all_seq_elmo_embedding.append(seq_elmo_embedding)
                                g_loop_r.append([uid, hit[1], query_seq, start_origin+1, stop_origin])

                            # probs_ = models[t].predict([all_seq_elmo_embedding, all_seq_transformed]) 
                            probs_ = models[t].predict(np.array(all_seq_transformed)) 

                            g_loop_r_p = pd.DataFrame(g_loop_r,columns=["Entry","Hit","Seq","Start","End"])             
                            g_loop_r_p['Score'] = probs_
                            g_loop_r_p = g_loop_r_p.sort_values(by='Score', ascending=False)
                            path_output_G_loop = outdir_pred + '/%s.txt' % t
                            g_loop_r_p.to_csv(path_output_G_loop, sep="\t",index=False)
                        
                            g_loop_r_p['Type'] = t
                            ps_results.append(g_loop_r_p)
                        else:
                            g_loop_r = []
                            g_loop_r_p = pd.DataFrame(g_loop_r,columns=["Entry","Hit","Seq","Start","End","Score"]) 
                            path_output_G_loop = outdir_pred + '/%s.txt' % t
                            g_loop_r_p.to_csv(path_output_G_loop, sep="\t",index=False)

                        print('%s prediction done...' % t)

                    else:
                        
                        AH_loop_r = [] 
                        all_seq_transformed = []
                        # all_seq_transformed, all_seq_elmo_embedding = [], []
                        AH_pred_infor = ID_type_peptides_position[uid][t]
                        
                        for g in AH_pred_infor:
                            hit = g.split('_')
                            start_origin = int(hit[1])-1
                            stop_origin = int(hit[2])
                            start = start_origin + shift
                            stop = stop_origin + shift
                            median_pos = (start+stop-1)//2
                            slice_start = median_pos - slicesize // 2
                            slice_stop = slice_start + slicesize
                            query_seq = protein_pad_local[slice_start:slice_stop]
                            seq_transformed = AA_encoding([query_seq])
                            all_seq_transformed.append(seq_transformed)
                            # seq_elmo_embedding = protein_pad_global[slice_start:slice_stop]
                            # all_seq_elmo_embedding.append(seq_elmo_embedding)
                            AH_loop_r.append([uid, hit[0], query_seq, start_origin+1, stop_origin])

                        # probs_ = models[t].predict([all_seq_elmo_embedding, all_seq_transformed]) 
                        probs_ = models[t].predict(np.array(all_seq_transformed)) 
                        
                        AH_loop_r_p = pd.DataFrame(AH_loop_r,columns=["Entry","Hit","Seq","Start","End"])             
                        AH_loop_r_p['Score'] = probs_
                        AH_loop_r_p = AH_loop_r_p.sort_values(by='Score', ascending=False)
                        path_output_AH_loop = outdir_pred + '/%s.txt' % t
                        AH_loop_r_p.to_csv(path_output_AH_loop, sep="\t",index=False)
                        
                        AH_loop_r_p['Type'] = t                
                        ps_results.append(AH_loop_r_p)
                        
                        print('%s prediction done...' % t)             
  
    ps_results1 = pd.concat(ps_results, ignore_index=True)
    ps_results1["DEGRON"] = "Known-INTERNAL"
    ps_results1["Entry_Isoform"] = ps_results1['Entry']
    ps_results1 = ps_results1.rename(columns = {'Start':'START','End':'END'})

def main(args):
    fasta_file = args.inputfile
    pred_type = args.motif
    out_file = args.outfile
    mod = args.fast
    if mod:
        pred_and_write_metrics_datatable_sel_mod(fasta_file, out_file, pred_type, both_embeddings="yes")

    else:
        pred_and_write_metrics_datatable_sel_mod(fasta_file, out_file, pred_type)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Drug region prediction by deep learning")
    parser.add_argument('-inf', '--inputfile', type=str,
                        help='One file containing predicted sequences.')
    parser.add_argument('-mof', '--motif', type=str, nargs='+',
                        help='DR hotspot type, spaces separated if more than one.')  
    parser.add_argument('-out', '--outfile', type=str, 
                    help='The output of the predicted result.') 
    parser.add_argument('-f', "--fast", action='store_true', help='run on comprehasive models')
                     
    args = parser.parse_args()
    
    main(args)
    
    # python /public/home/hxu6/projects/Web_kinDR/github/20250306/DrKinase_prediction.py -inf '/public/home/hxu6/projects/Web_kinDR/github/20250306/test/test.txt' -mof 'Gatekeeper' 'G-loop' 'A-loop' 'αC-Helix' -out '/public/home/hxu6/projects/Web_kinDR/github/20250306/prediction/'
    # python DrKinase_prediction.py -inf 'test/test.txt' -mof 'Gatekeeper' 'G-loop' 'A-loop' 'αC-Helix' -out 'prediction/'

