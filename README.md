# Compound-protein interactions

#### Description
Hypergraph-based dual-channel improved variational autoencoder with cross-attention for compound-protein interactions identification
This repository contains the PyTorch implementation of framework, as described in our paper "Hypergraph-based dual-channel improved variational autoencoder with cross-attention for compound-protein interactions identification". The framework is a dual-channel hypergraph model with improved variational autoencoder and cross-attention mechanism to identify potential compound-protein interactions. It works on hypergraph, PubChem fingerprint descriptors of compounds and primary structure features of proteins.

#### Dependencies
The source code developed in Python 3.10 using PyTorch 1.13.1+cu117. The required python dependencies are given below. 
Matlab = 2024b
pytorch = 1.13.1+cu117
dgh = 0.9.4
pandas = 2.3.1
numpy = 1.23.5
scipy = 1.15.3

#### Instructions
(1) Collect compound-protein interaction information, chemical SMILEs and protein sequence data from databases DrugBank and UniprotKB, and calculate molecular fingerprint descriptors and protein primary structure features.
(2) Construct a hypergraph with compounds as vertices and proteins as hyperedges, and a hypergraph with compounds as hyperedges and proteins as vertices, based on collected compound-protein interaction data. 
(3) Two hypergraph structures and corresponding vertex feature matrices are input into the improved hypergraph variational autoencoder, and the obtained embedded features are used to represent compounds and proteins, respectively.
(4) Multi-head cross-attention operations are performed on the embedded features of proteins and compounds to obtain fusion features that capture their interaction information. Then, the fusion features are fed into deep neural network model to identify potential compound-protein interactions.  

#### Methods
1. Run HyperGraphVAEDrugEdgeGPU.py
2. Run HyperGraphVAEProteinEdgeGPU.py
3. Run CrossAttentionModel.m


