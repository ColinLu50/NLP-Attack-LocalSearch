import numpy as np

def check_attack_result(x_adv, x_orig, target_label, attacker):
    if x_adv is not None:
        # check result is correct
        tmp_pred = attacker.clean_predict(x_adv)
        if np.argmax(tmp_pred) != target_label:
            raise Exception('search result is not reach target')

        # check change words
        num_changes = np.sum(x_orig != x_adv)
        if num_changes == 0:
            raise Exception('search result is same as raw')