import os
import sys
import json
import math
import random
import numpy as np
from scipy.special import logsumexp
import matplotlib.pyplot as plt

class EXPLearner:
    def __init__(self, lambda_, T, threshold_candidates=np.arange(0, 1, 0.01)):
        self.thresholds = threshold_candidates
        self.lambda_ = lambda_
        self.T = T

        self.weights = np.ones(len(self.thresholds)) / len(self.thresholds)
        self.losses = np.zeros(len(self.thresholds))

    def predict(self):
        idx = np.random.choice(len(self.thresholds), p=self.weights)
        return self.thresholds[idx], idx
    
    def update(self, IDK_flag, contradiction_flag, idx, alpha):
        lr = math.sqrt(
            math.log(len(self.thresholds)) /
            (2 * ((self.lambda_ + 1) ** 2) * self.T * len(self.thresholds))
        )

        ineff = IDK_flag
        f_gap = alpha if IDK_flag == 1 else float(contradiction_flag)
        self.losses = np.full(len(self.thresholds), -(self.lambda_ + 1))
        self.losses[idx] += (ineff + self.lambda_ * f_gap + (self.lambda_ + 1)) / self.weights[idx]
        
        self.weights *= np.exp(-lr * self.losses - logsumexp(-lr * self.losses))
        self.weights /= np.sum(self.weights)
        
def online_learning(question_dict, lambda_, alpha, T):
    learner = EXPLearner(
        threshold_candidates=np.arange(0, 1, 0.01),
        lambda_=lambda_,
        T=T
    )
    
    records = []
    total_questions = 0
    non_idk_count = 0
    non_idk_contradict_count = 0

    if isinstance(question_dict, dict):
        questions = list(question_dict.values())
    else:
        questions = question_dict.copy()
    random.shuffle(questions)
    
    for entry in questions:
        total_questions += 1
        
        question = entry.get("question")
        response_text = entry.get("response_text")
        answer = entry.get("answer")
        confidence_score = entry.get("current_score")
        contradiction_flag = entry.get("contradition_flag")
        
        threshold, idx = learner.predict()
        IDK_flag = 1 if confidence_score < threshold else 0
        
        learner.update(IDK_flag, contradiction_flag, idx, alpha)
        
        if IDK_flag == 0:
            non_idk_count += 1
            if contradiction_flag == 1:
                non_idk_contradict_count += 1
                
        records.append({
            "question": question,
            "response_text": response_text,
            "answer": answer,
            "confidence_score": confidence_score,
            "threshold": threshold,
            "IDK_flag": IDK_flag,
            "contradiction_flag": contradiction_flag
        })

    
    eff = non_idk_count / total_questions if total_questions > 0 else 0
    fdr_e = (non_idk_contradict_count / non_idk_count) if non_idk_count > 0 else 0
    
    print("Total_questions:", total_questions)
    print("Efficiency: {:.2f}".format(eff))
    print("FDR-E: {:.2f}".format(fdr_e))
    
    return records, eff, fdr_e

def run_experiments(alpha_values, lambda_, num_trials, T):
    with open("test_data/nq_paired_dict.json", "r", encoding="utf-8") as f:
        question_dict = json.load(f)
        
    efficiency_dict = {alpha: [] for alpha in alpha_values}
    fdr_e_dict = {alpha: [] for alpha in alpha_values}

    for alpha in alpha_values:
        print(f"\nRunning online_learning for lambda = {lambda_}, alpha = {alpha:.2f}")
        for _ in range(num_trials):
            records, eff, fdr_e = online_learning(question_dict, lambda_, alpha, T=T)
            efficiency_dict[alpha].append(eff)
            fdr_e_dict[alpha].append(fdr_e)
            
    return efficiency_dict, fdr_e_dict

def plot_violin_results(efficiency_dict, fdr_e_dict, save_path):
    alpha_list = sorted(efficiency_dict.keys())
    x = np.arange(len(alpha_list))

    eff_data = [efficiency_dict[a] for a in alpha_list]
    fdr_data = [fdr_e_dict[a] for a in alpha_list]

    fig, axs = plt.subplots(1, 2, figsize=(16, 8))

    # Efficiency plot
    axs[0].violinplot(eff_data, positions=x, widths=0.6, showmeans=True, showmedians=True)
    axs[0].set_title("Efficiency Distribution")
    axs[0].set_xticks(x)
    axs[0].set_xticklabels([f"{a:.2f}" for a in alpha_list])
    axs[0].set_xlabel("target error rate (α)")
    axs[0].set_ylabel("Efficiency")
    axs[0].yaxis.grid(True, linestyle='--', alpha=0.7)

    # FDR-E plot
    axs[1].violinplot(fdr_data, positions=x, widths=0.6, showmeans=True, showmedians=True)
    axs[1].set_title("FDR-E Distribution")
    axs[1].set_xticks(x)
    axs[1].set_xticklabels([f"{a:.2f}" for a in alpha_list])
    axs[1].set_xlabel("target error rate (α)")
    axs[1].set_ylabel("FDR-E")
    axs[1].yaxis.grid(True, linestyle='--', alpha=0.7)
    
    for xi, α in zip(x, alpha_list):
        axs[1].axhline(y=α, color='grey', linestyle='--', linewidth=1)

    fig.suptitle("Adversarial Bandit Results", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path)
    plt.close()
    print(f"plot saved to {save_path}")

if __name__ == "__main__":
    output_dir = f"adv_bandit_plot"
    os.makedirs(output_dir, exist_ok=True)
    
    alpha_values = np.arange(0.20, 0.22, 0.01).tolist()
    lambda_ = 200
    num_trials = 1
    T = 64330

    efficiency_dict, fdr_e_dict = run_experiments(alpha_values, lambda_, num_trials, T=T)
    save_path = os.path.join(output_dir, "adversarial_bandit_result.png")
    plot_violin_results(efficiency_dict, fdr_e_dict, save_path)
