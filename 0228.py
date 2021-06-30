import pandas as pd, numpy as np, re
import dill
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from contextualbandits.online import BootstrappedUCB, BootstrappedTS, SeparateClassifiers,\
            EpsilonGreedy, AdaptiveGreedy, ExploreFirst, ActiveExplorer, SoftmaxExplorer
from copy import deepcopy

from sklearn.linear_model import SGDClassifier

def parse_data(file_name):
    features = list()
    labels = list()
    with open(file_name, 'rt') as f:
        f.readline()
        for l in f:
            if bool(re.search("^[0-9]", l)):
                g = re.search("^(([0-9]{1,2},?)+)\s(.*)$", l)
                labels.append([int(i) for i in g.group(1).split(",")])
                features.append(eval("{" + re.sub("\s", ",", g.group(3)) + "}"))
            else:
                l = l.strip()
                labels.append([])
                features.append(eval("{" + re.sub("\s", ",", l) + "}"))
    features = pd.DataFrame.from_dict(features).fillna(0).iloc[:,:].values
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(labels)
    return features, y

X, y = parse_data("loc_1.txt")


nchoices = y.shape[1]
base_algorithm = SGDClassifier(random_state=123, loss='log')
beta_prior = ((3, 7), 2) # until there are at least 2 observations of each class, will use prior Beta(3, 7)

## The base algorithm is embedded in different metaheuristics
bootstrapped_ucb = BootstrappedUCB(deepcopy(base_algorithm), nchoices = nchoices, beta_prior=beta_prior, batch_train=True)
bootstrapped_ts = BootstrappedTS(deepcopy(base_algorithm), nchoices = nchoices, beta_prior=beta_prior, batch_train=True)
one_vs_rest = SeparateClassifiers(deepcopy(base_algorithm), nchoices = nchoices, beta_prior=beta_prior, batch_train=True)
epsilon_greedy = EpsilonGreedy(deepcopy(base_algorithm), nchoices = nchoices, beta_prior=beta_prior, batch_train=True)
epsilon_greedy_nodecay = EpsilonGreedy(deepcopy(base_algorithm), nchoices = nchoices,
                                     beta_prior=beta_prior, decay=None, batch_train=True)
adaptive_greedy_thr = AdaptiveGreedy(deepcopy(base_algorithm), nchoices=nchoices,
                                     decay_type='threshold', batch_train=True)
adaptive_greedy_perc = AdaptiveGreedy(deepcopy(base_algorithm), nchoices = nchoices, beta_prior=beta_prior,
                                      decay_type='percentile', decay=0.9997, batch_train=True)
explore_first = ExploreFirst(deepcopy(base_algorithm), nchoices = nchoices,
                             beta_prior=None, explore_rounds=1500, batch_train=True)
active_explorer = ActiveExplorer(deepcopy(base_algorithm), nchoices = nchoices, beta_prior=beta_prior, batch_train=True)
adaptive_active_greedy = AdaptiveGreedy(deepcopy(base_algorithm), nchoices = nchoices, beta_prior=beta_prior,
                                        active_choice='weighted', decay_type='percentile', decay=0.9997, batch_train=True)
softmax_explorer = SoftmaxExplorer(deepcopy(base_algorithm), nchoices = nchoices, beta_prior=beta_prior, batch_train=True)

models = [bootstrapped_ucb, bootstrapped_ts, one_vs_rest, epsilon_greedy, epsilon_greedy_nodecay,
          adaptive_greedy_thr, adaptive_greedy_perc, explore_first, active_explorer,
          adaptive_active_greedy, softmax_explorer]

# These lists will keep track of the rewards obtained by each policy
rewards_ucb, rewards_ts, rewards_ovr, rewards_egr, rewards_egr2, \
rewards_agr, rewards_agr2, rewards_efr, rewards_ac, \
rewards_aac, rewards_sft = [list() for i in range(len(models))]

lst_rewards = [rewards_ucb, rewards_ts, rewards_ovr, rewards_egr, rewards_egr2,
               rewards_agr, rewards_agr2, rewards_efr, rewards_ac,
               rewards_aac, rewards_sft]

# batch size - algorithms will be refit after N rounds
batch_size=500

# initial seed - all policies start with the same small random selection of actions/rewards
first_batch = X[:batch_size, :]
action_chosen = np.random.randint(nchoices, size=batch_size)
rewards_received = y[np.arange(batch_size), action_chosen]

# fitting models for the first time
for model in models:
    np.random.seed(123)
    model.fit(X=first_batch, a=action_chosen, r=rewards_received)
    
# these lists will keep track of which actions does each policy choose
lst_a_ucb, lst_a_ts, lst_a_ovr, lst_a_egr, lst_a_egr2, lst_a_agr, \
lst_a_agr2, lst_a_efr, lst_a_ac, lst_a_aac, \
lst_a_sft = [action_chosen.copy() for i in range(len(models))]

lst_actions = [lst_a_ucb, lst_a_ts, lst_a_ovr, lst_a_egr, lst_a_egr2, lst_a_agr,
               lst_a_agr2, lst_a_efr, lst_a_ac, lst_a_aac,lst_a_sft]

# rounds are simulated from the full dataset
def simulate_rounds_stoch(model, rewards, actions_hist, X_batch, y_batch, rnd_seed):
    np.random.seed(rnd_seed)
    
    ## choosing actions for this batch
    actions_this_batch = model.predict(X_batch).astype('uint8')
    
    # keeping track of the sum of rewards received
    rewards.append(y_batch[np.arange(y_batch.shape[0]), actions_this_batch].sum())
    
    # adding this batch to the history of selected actions
    new_actions_hist = np.append(actions_hist, actions_this_batch)
    
    # rewards obtained now
    rewards_batch = y_batch[np.arange(y_batch.shape[0]), actions_this_batch]
    
    # now refitting the algorithms after observing these new rewards
    np.random.seed(rnd_seed)
    model.partial_fit(X_batch, actions_this_batch, rewards_batch)
    
    return new_actions_hist

# now running all the simulation
for i in range(int(np.floor(X.shape[0] / batch_size))):
    batch_st = (i + 1) * batch_size
    batch_end = (i + 2) * batch_size
    batch_end = np.min([batch_end, X.shape[0]])
    
    X_batch = X[batch_st:batch_end, :]
    y_batch = y[batch_st:batch_end, :]
    
    for model in range(len(models)):
        lst_actions[model] = simulate_rounds_stoch(models[model],
                                                   lst_rewards[model],
                                                   lst_actions[model],
                                                   X_batch, y_batch,
                                                   rnd_seed = batch_st)

#    for model in range(len(models)):
#    	dill.dump(models[model], open("model_%d_loc1.dill" % (model), "wb"))

        
#plotting

import matplotlib.pyplot as plt
from pylab import rcParams

def get_mean_reward(reward_lst, batch_size=batch_size):
    mean_rew=list()
    for r in range(len(reward_lst)):
        mean_rew.append(sum(reward_lst[:r+1]) * 1.0 / ((r+1)*batch_size))
    return mean_rew


import scipy.stats as st
y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11 = get_mean_reward(rewards_ucb), get_mean_reward(rewards_ts) \
,get_mean_reward(rewards_ovr), get_mean_reward(rewards_egr), get_mean_reward(rewards_egr2) \
,get_mean_reward(rewards_agr), get_mean_reward(rewards_agr2), get_mean_reward(rewards_efr) \
,get_mean_reward(rewards_ac), get_mean_reward(rewards_aac), get_mean_reward(rewards_sft)
x1, x2 = [index for index in range(len(rewards_ucb))], [index for index in range(len(rewards_ts))]
x3, x4 = [index for index in range(len(rewards_ovr))], [index for index in range(len(rewards_egr))]
x5, x6 = [index for index in range(len(rewards_egr2))], [index for index in range(len(rewards_agr))]
x7, x8 = [index for index in range(len(rewards_agr2))], [index for index in range(len(rewards_efr))]
x9, x10 = [index for index in range(len(rewards_ac))], [index for index in range(len(rewards_aac))]
x11 = [index for index in range(len(rewards_sft))]


def CI_model(y, confidence = 0.95):
    std_err_y = st.sem(y1)
    n_y = len(y1)
    h_y = std_err_y * st.t.ppf((1 + confidence) / 2, n_y - 1)
    return h_y

h_y1, h_y2, h_y3, h_y4, h_y5, h_y6, h_y7, h_y8, h_y9, h_y10, h_y11 = CI_model(y1), CI_model(y2), CI_model(y3),\
CI_model(y4), CI_model(y5), CI_model(y6), CI_model(y7), CI_model(y8), CI_model(y9), CI_model(y10), CI_model(y11)

ax = plt.subplot(111)

plt.errorbar(x1, y1, yerr= h_y1, label='Bootstrapped Upper-Confidence Bound (C.I.=80%)')
plt.errorbar(x2, y2, yerr= h_y2, label='Bootstrapped Thompson Sampling')
plt.errorbar(x3, y3, yerr= h_y3, label='Separate Classifiers + Beta Prior')
plt.errorbar(x4, y4, yerr= h_y4, label='Epsilon-Greedy (p0=20%, decay=0.9999')
plt.errorbar(x5, y5, yerr= h_y5, label='Epsilon-Greedy (p0=20%, no decay')
plt.errorbar(x6, y6, yerr= h_y6, label='Adaptive Greedy (decaying threshold)')
plt.errorbar(x7, y7, yerr= h_y7, label='Adaptive Greedy (p0=30%, decaying percentile)')
plt.errorbar(x8, y8, yerr= h_y8, label='Explore First (n=1,500)')
plt.errorbar(x9, y9, yerr= h_y9, label='Active Explorer')
plt.errorbar(x10, y10, yerr= h_y10, label='Adaptive Active Greedy')
plt.errorbar(x11, y11, yerr= h_y11, label='Softmax Explorer')
plt.plot(np.repeat(y.mean(axis=0).max(),len(rewards_sft)),linewidth=4,ls='dashed', label='Overall Best Arm (no context)')


plt.xlabel('Rounds (models were updated every 50 rounds)', size=10)
plt.ylabel('Cummulative Mean Reward', size=10)
#plt.title('Comparison of Online Contextual Bandit Policies in location 7\n(Base Algorithm is Logistic Regression with data fit in streams)\n\nDataset\n(159 categories, 1836 attributes)',size=30)
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 1.25])
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, ncol=3, prop={'size':20})

plt.savefig("location_1.png", bbox_inches='tight', dpi = 600)