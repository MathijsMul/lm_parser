from arc_hybrid import Configuration
from utils import transition_from_code
import torch
import torch.nn as nn
from torch.autograd import Variable
from dictionary_corpus import tokenize

class Parser(nn.Module):
    """
    Shift-reduce parser inspired by Kiperwasser & Goldberg (2016, https://aclweb.org/anthology/Q16-1023),
    replacing bidirectional LSTM input features with the hidden states of Gulordava e.a.'s
    (2018, https://arxiv.org/abs/1803.11138) best pre-trained English language model

    Attributes:
        name: name of the model
        lm: pre-trained language model
        lm_dict: vocabulary dictionary of pre-trained language model
        hidden_units_mlp: dimensionality of hidden MLP (classification) layer
        arc_labels: list of arc labels
        arc_label_to_idx: dictionary from arc labels to indices
        arc_idx_to_label: inverse of the above dictionary
        num_transitions: number of different transitions (classes)
        lm_rnn_hidden_size: number of hidden units of pre-trained language model
        mlp_in: MLP input layer
        tanh: Tanh nonlinearity for MLP
        mlp_out: MLP output layer
        features: locations of feature items from arc hybrid configuration (stack & buffer)
    """

    def __init__(self, name, language_model, lm_dictionary, hidden_units_mlp, arc_labels, features):
        super(Parser, self).__init__()
        self.name = name
        self.lm = language_model

        # Freeze language model parameters
        for param in self.lm.parameters():
            param.requires_grad = False

        self.lm_dict = lm_dictionary
        self.hidden_units_mlp = hidden_units_mlp # dimensionality of MLP hidden units
        self.arc_labels = arc_labels
        self.arc_label_to_idx = {arc_label: idx + 1 for idx, arc_label in enumerate(self.arc_labels)}
        self.arc_idx_to_label = {idx + 1 : arc_label for idx, arc_label in enumerate(self.arc_labels)}
        self.num_transitions = 2 * len(arc_labels) + 1 # number of different shift/reduce actions (the labels)
        self.lm_rnn_hidden_size = self.lm.rnn.hidden_size
        self.mlp_in = nn.Linear(in_features = 4 * self.lm_rnn_hidden_size, out_features = self.hidden_units_mlp)  # MLP to-hidden matrix, assuming 4 features
        self.tanh = nn.Tanh() # MLP nonlinearity
        self.mlp_out = nn.Linear(in_features = self.hidden_units_mlp, out_features = self.num_transitions) # MLP to-output matrix

        if features == 'default':
            # def features:  top 3 items on the stack and the first item on the buffer
            self.features = {'stack' : [-3, -2, -1],
                             'buffer' : [0]}

    def forward(self, words, features=None, output_to_conll=False):
        length_sentence = len(words)
        lm_indices = tokenize(self.lm_dict, words).unsqueeze(1)
        lm_hidden_states = self.lm.get_hidden_states(lm_indices) # hidden initialized as zero matrix by default

        if output_to_conll:
            conll_output = ''

        if features is None:
            # Do configutation - transition - configuration one by one (sequentially)
            outputs = []
            c = Configuration([str(i) for i in range(length_sentence)])

            while not c.is_empty():
                configuration = c.extract_features(self.features)
                configuration_tensor = torch.cat([self.get_lm_representation(word_idx, lm_hidden_states) for word_idx in configuration], 1)
                mlp_output = self.classification_layers(configuration_tensor)

                top_indices = torch.topk(mlp_output, self.num_transitions)[1][0]
                for entry in top_indices.split(1):
                    transition = transition_from_code(entry.item(), self.arc_idx_to_label)
                    if c.transition_admissible(transition):
                        outputs += [entry.item()]
                        c.apply_transition(transition)
                        break

            if output_to_conll:
                conll_fragment = self.arcs_to_conll(c.arcs)
                conll_output += conll_fragment

        else:
            # During training the features (sequences of configurations) are given
            num_configurations = len(features)
            all_input_features = torch.zeros((num_configurations, 4 * self.lm_rnn_hidden_size))  # initialize container

            for idx_conf, configuration in enumerate(features):
                lm_tensors = [self.get_lm_representation(word_idx, lm_hidden_states) for word_idx in configuration]
                configuration_tensor = torch.cat(lm_tensors, 1)
                all_input_features[idx_conf,:] = configuration_tensor

            outputs = self.classification_layers(all_input_features)

        if output_to_conll:
            return(outputs, conll_output)
        else:
            return(outputs)

    def get_lm_representation(self, word_idx, lm_hidden_states):
        if word_idx is None:
            # Return zero tensor for emtpy feature positions
            return (torch.zeros((1, self.lm_rnn_hidden_size)))
        else:
            return (lm_hidden_states[int(word_idx)])

    def classification_layers(self, lm_features):
        mlp_hidden = self.mlp_in(lm_features)
        mlp_hidden_activated = self.tanh(mlp_hidden)
        mlp_output = self.mlp_out(mlp_hidden_activated)
        return(mlp_output)

    def arcs_to_conll(self, arcs):
        """
        Translate arcs for a sentence to CONLL fragment.

        :param arcs: Arcs object
        :return: CONLL fragment
        """

        conll_output = ''
        all_labeled_arcs = arcs.contents
        number_words = len(all_labeled_arcs)
        indices = [i + 1 for i in range(number_words)]

        for word_index in indices:
            for arc in all_labeled_arcs:
                if arc[1] == str(word_index):
                    conll_output += str(word_index) + '\t' + 'WORD' + '\t' + '_' + '\t' + 'TAG' + '\t' + 'TAG' + '\t' + '_' + '\t' + str(arc[0]) + '\t' + str(arc[2]) + '\t' + '_' + '\t' + '_' + '\n'
        return(conll_output)
