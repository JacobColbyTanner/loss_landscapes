import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import loss_landscape_functions


save_name = "/N/project/networkRNNs/loss_lndscape_data/four_tasks_all_weights_modules.npz"



TWJ = np.load(save_name)

data = TWJ["total_weight_traj"]



ndim = 3
max_corr = False

#data = TWJ["total_weight_traj"]
all_corr = loss_landscape_functions.pca_validate(data,ndim,max_corr)



import matplotlib.pyplot as plt

np.save("/N/project/networkRNNs/loss_lndscape_data/Four_tasks_modules_all_corr.npy", all_corr)


plt.plot(np.absolute(all_corr))
plt.xlabel("number of iterations")
plt.ylabel("similarity to other sample (N=500)")
plt.title("Four tasks no modules")
plt.savefig("/N/project/networkRNNs/loss_lndscape_data/Four_tasks_modules.png", dpi=300, format='png', bbox_inches='tight')
plt.show()
