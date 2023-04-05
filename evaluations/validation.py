import os

import torch
import numpy as np

from evaluations.fid import calculate_frechet_distance
from evaluations.prd import compute_prd_from_embedding, prd_to_max_f_beta_pair
# from vae_utils import generate_images
from scipy.stats import wasserstein_distance


class Validator:
    def __init__(self, n_classes, device, dataset, stats_file_name, dataloader, score_model_device=None,
                 multi_label_classifier=False):
        self.n_classes = n_classes
        self.device = device
        if not score_model_device:
            score_model_device = device
        self.dataset = dataset
        self.score_model_device = score_model_device
        self.dataloader = dataloader
        self.multi_label_classifier = multi_label_classifier
        if multi_label_classifier:
            self.classifier_loss = torch.nn.BCEWithLogitsLoss()
        else:
            self.classifier_loss = torch.nn.CrossEntropyLoss()

        print("Preparing validator")
        if dataset in ["MNIST", "Omniglot"]:  # , "DoubleMNIST"]:
            if dataset in ["Omniglot"]:
                from evaluations.evaluation_models.lenet_Omniglot import Model
            else:
                from evaluations.evaluation_models.lenet import Model
            net = Model()
            model_path = "evaluations/evaluation_models/lenet_" + dataset
            net.load_state_dict(torch.load(model_path))
            net.to(device)
            net.eval()
            self.dims = 128 if dataset in ["Omniglot", "DoubleMNIST"] else 84  # 128
            self.score_model_func = net.part_forward
        elif dataset.lower() in ["celeba", "doublemnist", "fashionmnist", "flowers", "cern", "cifar10", "lsun",
                                 "imagenet", "malaria", "cifar100", "birds", "svhn", "mnist32", "da_svhn_mnist",
                                 "office_a", "office_d", "office_w", "usps"]:
            from evaluations.evaluation_models.inception import InceptionV3
            self.dims = 2048
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[self.dims]
            model = InceptionV3([block_idx]).to(device)
            if score_model_device:
                model = model.to(score_model_device)
            model.eval()
            self.score_model_func = lambda batch: model(batch)[0]
        else:
            raise NotImplementedError
        self.stats_file_name = f"{stats_file_name}_dims_{self.dims}"

    @torch.no_grad()
    def calculate_accuracy_with_classifier(self, model):
        model.eval()
        test_loader = self.dataloader
        correct = 0
        total = 0
        test_loss = 0
        for idx, batch in enumerate(test_loader):
            x, cond = batch
            x = x.to(self.device)
            y = cond['y'].to(self.device)
            out_classifier = model.classify(x)
            if self.multi_label_classifier:
                preds = out_classifier > 0
                correct += (preds == y).sum()
                total += y.numel()
                y = y.float()
            else:
                preds = torch.argmax(out_classifier, 1)
                correct += (preds == y).sum()
                total += len(y)
            test_loss += self.classifier_loss(out_classifier, y)
        model.train()
        return preds, test_loss / idx, correct / total

    @torch.no_grad()
    def calculate_results(self, train_loop, n_generated_examples, dataset=None, batch_size=128):
        test_loader = self.dataloader
        distribution_orig = []
        distribution_gen = []

        precalculated_statistics = False
        os.makedirs(f"results/orig_stats/", exist_ok=True)
        stats_file_path = f"results/orig_stats/{self.dataset}_{self.stats_file_name}.npy"
        if os.path.exists(stats_file_path):
            print(f"Loading cached original data statistics from: {self.stats_file_name}")
            distribution_orig = np.load(stats_file_path)
            precalculated_statistics = True

        print("Calculating FID:")
        if not precalculated_statistics:
            print("Calculating original statistics")
            for idx, batch in enumerate(test_loader):
                x, cond = batch
                x = x.to(self.device)
                if dataset.lower() in ["fashionmnist", "doublemnist"]:
                    x = x.repeat([1, 3, 1, 1])
                distribution_orig.append(self.score_model_func(x.to(self.score_model_device)).cpu().detach().numpy())
                if idx % 10 == 0:
                    print(idx, len(test_loader))

            distribution_orig = np.array(np.concatenate(distribution_orig)).reshape(-1, self.dims)
            np.save(stats_file_path, distribution_orig)

        examples, _ = train_loop.generate_examples(total_num_exapmles=n_generated_examples,
                                                   max_classes=self.n_classes,
                                                   batch_size=batch_size)
        examples_to_generate = n_generated_examples
        i = 0
        while examples_to_generate > 0:
            print(examples_to_generate)
            example = examples[i * batch_size:min(n_generated_examples, (i + 1) * batch_size)].to(
                self.score_model_device)
            if dataset.lower() in ["fashionmnist", "doublemnist"]:
                example = example.repeat([1, 3, 1, 1])
            distribution_gen.append(self.score_model_func(example).cpu().detach())  # .numpy().reshape(-1, self.dims))
            examples_to_generate -= batch_size
            i += 1
            # distribution_gen = self.score_model_func(example).cpu().numpy().reshape(-1, self.dims)

        distribution_gen = torch.cat(distribution_gen).numpy().reshape(-1, self.dims)
        # distribution_gen = np.array(np.concatenate(distribution_gen)).reshape(-1, self.dims)

        precision, recall = compute_prd_from_embedding(
            eval_data=distribution_orig[np.random.choice(len(distribution_orig), len(distribution_gen), False)],
            ref_data=distribution_gen)
        precision, recall = prd_to_max_f_beta_pair(precision, recall)
        print(f"Precision:{precision},recall: {recall}")
        return calculate_frechet_distance(distribution_gen, distribution_orig), precision, recall
