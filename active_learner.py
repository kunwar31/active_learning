import numpy as np
from typing import List
from flair.data import Sentence
from flair.trainers import ModelTrainer
from flair.optim import AdamW


class ActiveLearner:
    def __init__(self,
                 sentences: List[Sentence],
                 experiment_name: str = "default_experiment_name",
                 data_path: str = "default_experiment_path",
                 oracle,
                 embeddings_storage_mode='cpu',
                 ):
        self.data_path = data_path
        self.sentences = sentences
        self.oracle = oracle
        self.experiment_name = experiment_name
        self.embeddings_storage_mode = embeddings_storage_mode

    def train_model(self, corpus, classifier, optimizer_state=None, epoch=1, lr=1e-5):
        trainer = ModelTrainer(classifier, corpus, optimizer=AdamW, optimizer_state=optimizer_state)
        result = trainer.train(f'{self.data_path}/{self.experiment_name}/',
                               learning_rate=lr,
                               min_learning_rate=1e-8,
                               mini_batch_size=32,
                               anneal_factor=0.5,
                               patience=5,
                               max_epochs=epoch,
                               embeddings_storage_mode=self.embeddings_storage_mode,
                               use_amp=True,
                               weight_decay=1e-4,)

        return classifier, result['optimizer_state']

    @staticmethod
    def get_predictions(data, classifier, mini_batch_size=1024):
        return [(s.labels[0].value,
                 s.labels[0].score) for s in classifier.predict(data, mini_batch_size=mini_batch_size)]

    def sample_confused_samples(self, classifier, labelling_step_size, sampling_multiplier, max_sample_size):
        train_data_sample_size = min(labelling_step_size * sampling_multiplier, max_sample_size)
        if len(self.sentences) > train_data_sample_size:
            random_indices = np.random.choice(a=len(self.sentences), size=train_data_sample_size, replace=False)
            sentence_sample = []
            for index in random_indices:
                sentence_sample.append(self.sentences[index])
        else:
            sentence_sample = self.sentences.copy()
            
        predictions = self.get_predictions(sentence_sample, classifier)
        weights = []
        for (label, pred) in predictions:
            weights.append(1 - abs(0.5-pred))
        weights = np.array(weights)
        weights = (weights * 2) - 1
        weights = weights / weights.sum()
        confused_idx = list(np.random.choice(a=len(predictions),
                                             size=labelling_step_size,
                                             p=weights,
                                             replace=False))
        return [sentence_sample[idx] for idx in confused_idx]

    def step(self,
             classifier,
             optimizer_state=None,
             labelling_step_size: int = 100,
             sampling_multiplier: int = 100,
             max_sample_size: int = 10000,
             step_lr=1e-5):

        confusion_samples = self.sample_confused_samples(classifier,
                                                         labelling_step_size,
                                                         sampling_multiplier, max_sample_size)
        corpus = self.oracle.get_labelled_corpus(confusion_samples, self.experiment_name)
        classifier, opt_state = self.train_model(corpus, classifier, optimizer_state, lr=step_lr)
        return classifier, opt_state





