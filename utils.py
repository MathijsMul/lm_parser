import subprocess

def test(model, datamanager_test_file, output_conll, batch_size=1):
    """
    Test model on data.

    :param model: BiLSTMParser object to test
    :param datamanager_test_file: ConllLoader for test data
    :param output_conll: location of CONLL containing data as returned by model
    :param batch_size: number of sentences to test per time,
        must be 1 because consecutive predictions depend on each other so parallelization is not possible at present
    :return: evaluation results
    """

    test_file = datamanager_test_file.file
    output_conll_file = open(output_conll, 'w')

    for idx, sentence in enumerate(datamanager_test_file.sentences_unshuffled):

        words = sentence['words']
        _, conll_fragment = model(words, output_to_conll=True)
        output_conll_file.write(conll_fragment)
        if idx != datamanager_test_file.num_samples - 1:
            output_conll_file.write('\n')

    output_conll_file.close()
    eval_results = eval(test_file, output_conll)
    return(eval_results)

def eval(conll_gold, conll_predicted):
    """
    Calls eval.pl to evaluate CONLL gold file against CONLL predicted by model.

    :param conll_gold: gold CONLL
    :param conll_predicted: predicted CONLL
    :return: (labeled_attachment_score, unlabeled_attachment_score, label_accuracy_score)
    """

    eval_output = subprocess.check_output(['perl',
                                 'eval.pl',
                                 '-q',
                                 '-g',
                                 conll_gold,
                                 '-s',
                                 conll_predicted]).decode("utf-8")
    s = eval_output.split('\n')
    labeled_attachment_score = float(s[0].split()[-2])
    unlabeled_attachment_score = float(s[1].split()[-2])
    label_accuracy_score = float(s[2].split()[-2])
    return(labeled_attachment_score, unlabeled_attachment_score, label_accuracy_score)

def transition_code(transition, arc_label_to_idx):
    """
    Encode transition.

    :param transition: 'shift', ('left', l) or ('right', l) for l some label
    :param arc_label_to_idx: dictionary mapping arc labels to indices
    :return: transition code
    """

    if transition == 'shift':
        return 0
    elif transition[0] == 'left':
        return(arc_label_to_idx[transition[1]])
    elif transition[0] == 'right':
        return (arc_label_to_idx[transition[1]] + len(arc_label_to_idx))

def transition_from_code(code, arc_idx_to_label):
    """
    Decode transition.

    :param code: transition code
    :param arc_idx_to_label: dictionary mapping indices to arc labels
    :return: transition 'shift', ('left', l) or ('right', l) for l some label
    """

    if code == 0:
        return('shift')
    else:
        if code in arc_idx_to_label:
            return(('left', arc_idx_to_label[code]))
        else:
            return (('right', arc_idx_to_label[code - len(arc_idx_to_label)]))

def crop_file(file_in, nr_sentences):
    """
    For development purposes: crop a data file to first n sentences.

    :param file_in: file path
    :param nr_sentences: n
    :return:
    """

    file_out = file_in.split('.')[-2] + str(nr_sentences) + '.conll'
    with open(file_in, 'r') as f_in:
        with open(file_out, 'w') as f_out:
            sentence_count = 0
            for idx, line in enumerate(f_in):
                if line == '\n':
                    sentence_count += 1
                    if sentence_count == nr_sentences:
                        break
                    else:
                        f_out.write(line)
                else:
                    f_out.write(line)