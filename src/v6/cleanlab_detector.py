'''
Created on Oct 18, 2021

@author: mkunjir 
'''

import cleanlab
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
# To silence convergence warnings caused by using a weak
# logistic regression classifier on image data
import warnings

from v6.detector import Detector

warnings.simplefilter("ignore")

class CL(Detector):

    def __init__(self, features, labels, params):    
        super(CL, self).__init__(features, labels, params)
        self.num_iter = 0
        self.confusion = params.get('confusion', True)

    def use_feedback(self, index2correct_label):
        for k,v in index2correct_label.items():
            self.labels[k] = v
    
    def detect_and_rank(self):
        # we use cleanlab to get the 
        # out-of-sample predicted probabilities using cross-validation
        # with a very simple, non-optimized logistic regression classifier
        #psx = cleanlab.latent_estimation.estimate_cv_predicted_probabilities(
        #     self.features, self.labels, clf=LogisticRegression(max_iter=1000, multi_class='auto',
        #     solver='lbfgs'), cv_n_folds=self.params['num_folds'])

        psx = cleanlab.latent_estimation.estimate_cv_predicted_probabilities(
             self.features, self.labels, clf=RandomForestClassifier(n_estimators=20, random_state=0,
             ), cv_n_folds=self.params['num_folds'])


        # STEP 1 - Compute confident joint

        # Verify inputs
        s = np.asarray(self.labels)
        psx = np.asarray(psx)

        # Find the number of unique classes if K is not given
        K = len(np.unique(s))

        # Estimate the probability thresholds for confident counting
        # You can specify these thresholds yourself if you want
        # as you may want to optimize them using a validation set.
        # By default (and provably so) they are set to the average class prob.
        thresholds = [np.mean(psx[:,k][s == k]) for k in range(K)] # P(s^=k|s=k)
        thresholds = np.asarray(thresholds)

        ## mayuresh: disabling the average probability thresholds and using the confusion matrix instead
        if self.confusion:
          thresholds = np.array([.5, .5]) ## We assume it's a binary classification!

        # Compute confident joint
        confident_joint = np.zeros((K, K), dtype = int)
        for i, row in enumerate(psx):
             s_label = s[i]
             # Find out how many classes each example is confidently labeled as
             confident_bins = row >= thresholds - 1e-6
             num_confident_bins = sum(confident_bins)
             # If more than one conf class, inc the count of the max prob class
             if num_confident_bins == 1:
                confident_joint[s_label][np.argmax(confident_bins)] += 1
             elif num_confident_bins > 1:
                confident_joint[s_label][np.argmax(row)] += 1

        # Normalize confident joint (use cleanlab, trust me on this)
        confident_joint = cleanlab.latent_estimation.calibrate_confident_joint(
             confident_joint, s)

        # STEP 2 - Find label errors

        # We arbitrarily choose at least 5 examples left in every class.
        # Regardless of whether some of them might be label errors.
        MIN_NUM_PER_CLASS = 5 
        # Leave at least MIN_NUM_PER_CLASS examples per class.
        # NOTE prune_count_matrix is transposed (relative to confident_joint)
        prune_count_matrix = cleanlab.pruning.keep_at_least_n_per_class(
             prune_count_matrix=confident_joint.T,
             n=MIN_NUM_PER_CLASS,
        )

        s_counts = np.bincount(s)
        noise_masks_per_class = []
        # For each row in the transposed confident joint
        for k in range(K):
          noise_mask = np.zeros(len(psx), dtype=bool)
          psx_k = psx[:, k]
          if s_counts[k] > MIN_NUM_PER_CLASS:  # Don't prune if not MIN_NUM_PER_CLASS
            for j in range(K):  # noisy label index (k is the true label index)
              if k != j:  # Only prune for noise rates, not diagonal entries
                num2prune = prune_count_matrix[k][j]
                if num2prune > 0:
                    # num2prune'th largest p(classk) - p(class j)
                    # for x with noisy label j
                    margin = psx_k - psx[:, j]
                    s_filter = s == j
                    threshold = -np.partition(
                        -margin[s_filter], num2prune - 1
                    )[num2prune - 1]
                    noise_mask = noise_mask | (s_filter & (margin >= threshold))
            noise_masks_per_class.append(noise_mask)
          else:
            noise_masks_per_class.append(np.zeros(len(s), dtype=bool))

        # Boolean label error mask
        label_errors_bool = np.stack(noise_masks_per_class).any(axis=0)

        # Remove label errors if given label == model prediction
        for i, pred_label in enumerate(psx.argmax(axis=1)):
          # np.all let's this work for multi_label and single label
          if label_errors_bool[i] and np.all(pred_label == s[i]):
            label_errors_bool[i] = False

        # Convert boolean mask to an ordered list of indices for label errors
        label_errors_idx = np.arange(len(s))[label_errors_bool]
        # self confidence is the holdout probability that an example
        # belongs to its given class label
        self_confidence = np.array(
          [np.mean(psx[i][s[i]]) for i in label_errors_idx]
        )
        margin = self_confidence - psx[label_errors_bool].max(axis=1)
        label_errors_idx = label_errors_idx[np.argsort(margin)]

        self.num_iter += 1
        return label_errors_idx
