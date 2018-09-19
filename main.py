from optparse import OptionParser
from data_loader import ConllLoader
from train import ModelTrainer
from parser import Parser
from dictionary_corpus import Dictionary
import torch

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--train", dest="train_file", help="CONLL train data file", default='data/train-stanford-raw.conll')
    parser.add_option("--test", dest="test_file", help="CONLL test data file", default=None)
    parser.add_option("--epochs", type="int", dest="num_epochs", help="Number of training epochs")
    parser.add_option("--hidden_units_mlp", type="int", dest="hidden_units_mlp", help="Nr of hidden MLP units", default=100)
    parser.add_option("--features", dest="features", help="Which features to select from stack-buffer configuration", default='default')
    parser.add_option("--criterion", dest="criterion", help="Loss function", default='CrossEntropy')
    parser.add_option("--optimizer", dest="optimizer", help="Optimizer algorithm", default='Adam')
    parser.add_option("--l2", type="float", dest="l2_penalty", help="L2 regularization term", default=0.0)
    parser.add_option("--alpha", type="float", dest="alpha", help="Parameter for word dropout probability", default=0.25)
    parser.add_option("--model_name", dest="model_name", help="Name of model")
    parser.add_option("--run", type="int", dest="run", help="Run index", default=1)
    parser.add_option("--language_model", dest="lm", help="Location of pre-trained language model whose hidden states are to be used as features.")
    parser.add_option("--train_directory", dest="train_dir", help="Location of pre-trained language model's training directory, containing vocabulary.")

    (options, args) = parser.parse_args()

    # Load train data
    datamanager_train_file = ConllLoader(input_file=options.train_file, oracle=True, alpha=options.alpha)
    datamanager_train_file.load_file()
    arc_labels = datamanager_train_file.arc_labels

    # If provided, load test data
    if not options.test_file is None:
        datamanager_test_file = ConllLoader(input_file=options.test_file, oracle=False)
        datamanager_test_file.load_file()
    else:
        datamanager_test_file = None

    lm_dictionary = Dictionary(options.train_dir)
    language_model = torch.load(options.lm, map_location='cpu') # can be changed to GPU if available
    language_model.eval()

    model = Parser(name=options.model_name,
                   language_model=language_model,
                   lm_dictionary=lm_dictionary,
                   hidden_units_mlp=options.hidden_units_mlp,
                   arc_labels=arc_labels,
                   features=options.features)

    print(model)

    trainer = ModelTrainer(model=model,
                           datamanager_train_file=datamanager_train_file,
                           datamanager_test_file=datamanager_test_file,
                           epochs=options.num_epochs,
                           criterion=options.criterion,
                           optimizer=options.optimizer,
                           run=options.run,
                           l2_penalty=options.l2_penalty)

    trainer.train(test_each_epoch=True)