from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from Chatbot_functions import *
import torch
from torch.jit import script, trace
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math
import json



USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
save_dir = os.path.join("/home/newuser/Downloads", "save")
# print(USE_CUDA)

writer_tb = SummaryWriter()

# Define path to new file
# corpus_name = "cornell movie-dialogs corpus"
# corpus = os.path.join("/home/newuser/Downloads", corpus_name)
# datafile = os.path.join(corpus, "formatted_movie_lines.txt")
#
#
# delimiter = '\t'
# # Unescape the delimiter
# delimiter = str(codecs.decode(delimiter, "unicode_escape"))
#
# # Initialize lines dict, conversations list, and field ids
# lines = {}
# conversations = []
# MOVIE_LINES_FIELDS = ["lineID", "characterID", "movieID", "character", "text"]
# MOVIE_CONVERSATIONS_FIELDS = ["character1ID", "character2ID", "movieID", "utteranceIDs"]
#
# # Load lines and process conversations
# print("\nProcessing corpus...")
# lines = loadLines(os.path.join(corpus, "movie_lines.txt"), MOVIE_LINES_FIELDS)
# print("\nLoading conversations...")
# conversations = loadConversations(os.path.join(corpus, "movie_conversations.txt"),
#                                   lines, MOVIE_CONVERSATIONS_FIELDS)
#
# # Write new csv file
# print("\nWriting newly formatted file...")
# with open(datafile, 'w', encoding='utf-8') as outputfile:
#     writer = csv.writer(outputfile, delimiter=delimiter, lineterminator='\n')
#     for pair in extractSentencePairs(conversations):
#         writer.writerow(pair)
#
# # Print a sample of lines
# print("\nSample lines from file:")
# printLines(datafile)
#
#
#

# voc, pairs = loadPrepareData(corpus, corpus_name, datafile, save_dir)
# # Print some pairs to validate
# print("\npairs:")
# for pair in pairs[:100]:
#     print(pair)
# print((corpus_name))

corpus_name = 'bridged'
voc, pairs = our_dataset(corpus_name)

print(pairs[0:100])
print('lala')
print(pairs[890])



# Trim voc and pairs
pairs = trimRareWords(voc, pairs, MIN_COUNT)

# Example for validation
small_batch_size = 5
batches = batch2TrainData(voc, [random.choice(pairs) for _ in range(small_batch_size)])
input_variable, lengths, target_variable, mask, max_target_len = batches

print("input_variable:", input_variable)
print("lengths:", lengths)
print("target_variable:", target_variable)
print("mask:", mask)
print("max_target_len:", max_target_len)

# # Configure models
model_name = 'cb_model'
attn_model = 'dot'
#attn_model = 'general'
#attn_model = 'concat'
hidden_size =1024
encoder_n_layers = 4
decoder_n_layers = 4
dropout = 0.3
batch_size = 64

# Set checkpoint to load from; set to None if starting from scratch
loadFilename = True
# loadFilename = True
checkpoint_iter = 1000
if loadFilename:
    loadFilename = os.path.join(save_dir, model_name, corpus_name,
                           '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
                           '{}_checkpoint.tar'.format(checkpoint_iter))


# Load model if a loadFilename is provided
if loadFilename:
    # If loading on same machine the model was trained on
    checkpoint = torch.load(loadFilename)
    # If loading a model trained on GPU to CPU
    #checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    voc.__dict__ = checkpoint['voc_dict']


print('Building encoder and decoder ...')
# Initialize word embeddings
embedding = nn.Embedding(voc.num_words, hidden_size)
# embedding.load_state_dict(torch.load('/home/newuser/Downloads/embedding.pth'))
# Initialize encoder & decoder models
encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
# if loadFilename:
#     embedding.load_state_dict(embedding_sd)
#     encoder.load_state_dict(encoder_sd)
#     decoder.load_state_dict(decoder_sd)
# Use appropriate device
# encoder = encoder.to(device)
# decoder = decoder.to(device)
print('Models built and ready to go!')

# Configure training/optimization
clip = 50.0
teacher_forcing_ratio = 1.0
learning_rate = 0.0001
decoder_learning_ratio = 5.0
n_iteration = 10000
print_every = 1
save_every = 1000

# # # Ensure dropout layers are in train mode
# encoder.train()
# decoder.train()

# Initialize optimizers
# print('Building optimizers ...')
# encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
# decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
# if loadFilename:
#     encoder_optimizer.load_state_dict(encoder_optimizer_sd)
#     decoder_optimizer.load_state_dict(decoder_optimizer_sd)

# # If you have cuda, configure cuda to call
# for state in encoder_optimizer.state.values():
#     for k, v in state.items():
#         if isinstance(v, torch.Tensor):
#             state[k] = v.cuda()
#
# for state in decoder_optimizer.state.values():
#     for k, v in state.items():
#         if isinstance(v, torch.Tensor):
#             state[k] = v.cuda()


# Run training iterations
# print("Starting Training!")
# trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
#            embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size,
#            print_every, save_every, clip, corpus_name, loadFilename, writer_tb)



# # Saving the trained model
# torch.save(encoder.state_dict(), '/home/newuser/Downloads/encoder.pth')
# torch.save(decoder.state_dict(), '/home/newuser/Downloads/decoder.pth')
# torch.save(embedding.state_dict(),'/home/newuser/Downloads/embedding.pth')
# #
# Load trained model
# embedding.load_state_dict(torch.load('/home/newuser/Downloads/embedding.pth'))
encoder.load_state_dict(torch.load('/home/newuser/Downloads/encoder.pth'))
decoder.load_state_dict(torch.load('/home/newuser/Downloads/decoder.pth'))


torch.save(encoder.state_dict(), '/home/newuser/Downloads/encoder_best.pth')
torch.save(decoder.state_dict(), '/home/newuser/Downloads/decoder_best.pth')
# embedding.save(torch.load('/home/newuser/Downloads/embedding_best.pth'))

# Set dropout layers to eval mode
encoder.eval()
decoder.eval()
#
# # Initialize search module
searcher = GreedySearchDecoder(encoder, decoder)

# Begin chatting (uncomment and run the following line to begin)
evaluateInput(encoder, decoder, searcher, voc)