import math
import sys
from copy import deepcopy

import gpytorch
import numpy as np
import torch
from torch.quasirandom import SobolEngine

from .gp import train_gp
from .utils import latin_hypercube

from icecream import ic

class TurboStepwise:
    def __init__(self, n_trust_regions, n_init, batch_size, num_dims, failure_tolerance, success_tolerance, trust_region_length_init, trust_region_length_min, trust_region_length_max):

        self.n_trust_regions = n_trust_regions
        self.batch_size = batch_size
        self.n_init = n_init
        self.num_dims = num_dims

        self.batch_no = 0

        # Save the full history
        self.trials = []

        self.n_cand = min(100 * self.num_dims, 5000)

        # Always assume [0,1] for each parameter
        self.lb = 0
        self.ub = 1

        # Create trust regions
        self.trust_regions = []
        for i in range(n_trust_regions):
            tr = TrustRegion(i, self, batch_size, num_dims, failure_tolerance, success_tolerance, trust_region_length_init, trust_region_length_min, trust_region_length_max)
            self.trust_regions.append(tr)

        self.trust_regions_deactivated = []

        # Very basic input checks
        #assert n_trust_regions > 1

    def _suggest_initial_points(self, trust_region):
        trials_tr = trust_region.create_random_trials(self.n_init, self.batch_no)
        return trials_tr

    def suggest_next_batch(self):

        self.batch_no += 1

        trial_batch = []

        # Handle active trust regions (not recently restarted)
        active_trs = self.active_trust_regions()

        if len(active_trs) > 0:
            x_cand_all = np.zeros((len(active_trs), self.n_cand, self.num_dims))
            y_cand_all = np.zeros((len(active_trs), self.n_cand, self.batch_size))

            for i, tr in enumerate(active_trs):
                x_cand, y_cand, hypers = tr.create_candidates(self.n_cand)

                x_cand_all[i] = x_cand
                y_cand_all[i] = y_cand

            x_next, tr_next = self.select_candidates(x_cand_all, y_cand_all)

            # Turn into trials
            for i, x in enumerate(x_next):
                tr = active_trs[tr_next[i][0]]
                trial = tr.create_trial(x, self.batch_no, "ts")
                trial_batch.append(trial)

        # Handle restarted trust regions
        inactive_trs = self.inactive_trust_regions()
        for tr in inactive_trs:
            random_trials = self._suggest_initial_points(tr)
            trial_batch = np.append(trial_batch, random_trials)
     
        # This batch includes trials from random inits and and can therefore
        # exceed the batch size
        return trial_batch
   
    def update_trust_regions(self, completed_trials):
        for tr in self.trust_regions:
            tr.update(completed_trials)

        # Add the trials to the global collection
        self.trials = np.append(self.trials, completed_trials)

    def select_candidates(self, x_cand, y_cand):
        active_trs = self.active_trust_regions()
        assert x_cand.shape == (len(active_trs), self.n_cand, self.num_dims)
        assert y_cand.shape == (len(active_trs), self.n_cand, self.batch_size)
        assert (
            x_cand.min() >= 0.0 and x_cand.max() <= 1.0 and np.all(np.isfinite(y_cand))
        )

        x_next = np.zeros((self.batch_size, self.num_dims))
        idx_next = np.zeros((self.batch_size, 1), dtype=int)

        for k in range(self.batch_size):
            i, j = np.unravel_index(
                np.argmin(y_cand[:, :, k]), (len(active_trs), self.n_cand)
            )
            assert y_cand[:, :, k].min() == y_cand[i, j, k]
            x_next[k, :] = deepcopy(x_cand[i, j, :])
            idx_next[k, 0] = i
            assert np.isfinite(
                y_cand[i, j, k]
            )  # Just to make sure we never select nan or inf

            # Make sure we never pick this point again
            y_cand[i, j, :] = np.inf

        return x_next, idx_next

    # Get all trust regions that are currently active
    def active_trust_regions(self):
        active_trs = []
        for tr in self.trust_regions:
            if len(tr.trials) > 0:
                active_trs.append(tr)
        return active_trs

    # Get the trust regions that have not been initialized yet
    def inactive_trust_regions(self):
        inactive_trs = []
        for tr in self.trust_regions:
            if len(tr.trials) == 0:
                inactive_trs.append(tr)
        return inactive_trs

    def get_best_trial(self):
        best_y = np.inf
        best_trial = None
        for t in self.trials:
            if t.y < best_y:
                best_y = t.y
                best_trial = t

        return best_trial

    def get_num_trials(self):
        return len(self.trials)

    def get_all_trials(self):
        return self.trials

    def data(self):
        data = {}
        data["trials"] = []

        for i, t in enumerate(self.trials, start=1):
            trial_data = t.data()
            trial_data["global_id"] = i
            data["trials"].append(trial_data)
        
        data["TuRBO_Hyperparameters"] = {}

        data["TuRBO_Hyperparameters"]["n_trust_regions"] = self.n_trust_regions
        data["TuRBO_Hyperparameters"]["batch_size"] = self.batch_size
        data["TuRBO_Hyperparameters"]["n_init"] = self.n_init
        data["TuRBO_Hyperparameters"]["num_dims"] = self.num_dims
        data["TuRBO_Hyperparameters"]["n_cand"] = self.n_cand

        ic(data["TuRBO_Hyperparameters"]["n_trust_regions"])
        return data

    def get_status(self):
        # print the number of point in each trust region
        num_points = []
        for tr in self.trust_regions:
            num_trials_tr = tr.get_num_trials()
            num_points.append(num_trials_tr)
            print(f"TR-{tr.name()}: {num_trials_tr}")

        return num_points

class TrustRegion:
    def __init__(self, id, turbo_instance, batch_size, num_dims, failure_tolerance, success_tolerance, trust_region_length_init, trust_region_length_min, trust_region_length_max, num_restarts=0) -> None:

        self.turbo = turbo_instance
        self.id = id
        self.num_restarts = num_restarts
        self.batch_size = batch_size
        self.num_dims = num_dims

        # Tolerances and counters
        self.success_tolerance = success_tolerance
        self.failure_tolerance = failure_tolerance

        self.trial_counter = 1
        self.trials = []

        # Trust region sizes
        self.length_min = trust_region_length_min
        self.length_max = trust_region_length_max
        self.length_init = trust_region_length_init

        # Keep track of the bounds for this TR
        self.lb = np.empty((0, 1))
        self.ub = np.empty((0, 1))
        self.weights = np.empty((0, 1))

        self.center_x = np.empty((0, 1))
        self.center_y = None

        self.failcount = 0
        self.succcount = 0
        self.length = self.length_init

        self.n_training_steps = 30

        # Remember the hypers for trust regions we don't sample from
        self.hypers = {}
        self.min_cuda = 1024
        self.max_cholesky_size = 2000
        self.use_ard = True

        self.device = "cpu"
        self.dtype = "float64"

    def create_random_trials(self, n_init, batch_no):
        x = latin_hypercube(n_init, self.num_dims)
        random_trials = []
        for xi in x:
            trial = Trial(self.trial_counter, self, "latin")
            self.trial_counter += 1
            trial.set_x(xi)
            trial.set_batch_no(batch_no)
            random_trials.append(trial)

        return random_trials

    # Move TR to archive and create new TR
    def restart(self):

        print(f"Replacing trust region {self.name()} with a new one")

        # Create a new TR
        new_tr = TrustRegion(
            self.id, self.turbo, self.batch_size, self.num_dims, self.failure_tolerance, self.success_tolerance, self.length_init, self.length_min, self.length_max, self.num_restarts + 1
        )
        old_tr = self.turbo.trust_regions[self.id]

        self.turbo.trust_regions[self.id] = new_tr
        self.turbo.trust_regions_deactivated.append(old_tr)

    def create_trial(self, x, batch_no, type="ts"):
        trial = Trial(self.trial_counter, self, type)
        trial.set_x(x)
        trial.set_batch_no(batch_no)
        self.trial_counter += 1

        # self.trials.append(trial)
        return trial

    def create_candidates(self, n_cand):
        # Don't retrain the model if no points were adeed
        n_training_steps = 0 if self.hypers else self.n_training_steps

        # Create x array
        x = np.zeros((0, self.num_dims))
        y = np.zeros((0, 1))
        for t in self.trials:
            if t.status == "completed":
                y = np.append(y, t.y)
                x = np.vstack((x, t.x))

        self.center_y = np.amin(y)

        # Standardize function values
        mu, sigma = np.median(y), y.std()
        sigma = 1.0 if sigma < 1e-6 else sigma
        y = (deepcopy(y) - mu) / sigma

        # Figure out what device we are running on
        if len(x) < self.min_cuda:
            device, dtype = torch.device("cpu"), torch.float64
        else:
            device, dtype = self.device, self.dtype

        # We use CG + Lanczos for training if we have enough data
        with gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
            x_torch = torch.tensor(x).to(device=device, dtype=dtype)
            y_torch = torch.tensor(y).to(device=device, dtype=dtype)
            # print(f"Training gp for tr-{self.id} using {len(self.trials)} points and {n_training_steps} training steps")
            gp = train_gp(
                train_x=x_torch,
                train_y=y_torch,
                use_ard=self.use_ard,
                num_steps=n_training_steps,
                hypers=self.hypers,
            )

            # Save state dict
            self.hypers = gp.state_dict()

        # Create the trust region boundaries
        x_center = x[y.argmin().item(), :][None, :]
        self.center_x = x_center[0]
        

        weights = gp.covar_module.base_kernel.lengthscale.cpu().detach().numpy().ravel()
        weights = weights / weights.mean()  # This will make the next line more stable
        weights = weights / np.prod(
            np.power(weights, 1.0 / len(weights))
        )  # We now have weights.prod() = 1
        
        lb = np.clip(x_center - weights * self.length / 2.0, 0.0, 1.0)
        ub = np.clip(x_center + weights * self.length / 2.0, 0.0, 1.0)

        # Set the bound as fields
        self.lb = lb[0]
        self.ub = ub[0]
        self.weights = weights

        # Draw a Sobolev sequence in [lb, ub]
        seed = np.random.randint(int(1e6))
        sobol = SobolEngine(self.num_dims, scramble=True, seed=seed)
        pert = sobol.draw(n_cand).to(dtype=dtype, device=device).cpu().detach().numpy()
        pert = lb + (ub - lb) * pert

        # Create a perturbation mask
        prob_perturb = min(20.0 / self.num_dims, 1.0)

        mask = np.random.rand(n_cand, self.num_dims) <= prob_perturb
        ind = np.where(np.sum(mask, axis=1) == 0)[0]
        mask[ind, np.random.randint(0, self.num_dims - 1, size=len(ind))] = 1

        # Create candidate points
        x_cand = x_center.copy() * np.ones((n_cand, self.num_dims))
        x_cand[mask] = pert[mask]

        # Figure out what device we are running on
        if len(x_cand) < self.min_cuda:
            device, dtype = torch.device("cpu"), torch.float64
        else:
            device, dtype = torch.device("cpu"), torch.float64
            # device, dtype = self.device, self.dtype

        # We may have to move the GP to a new device
        gp = gp.to(dtype=dtype, device=device)

        # We use Lanczos for sampling if we have enough data
        with torch.no_grad(), gpytorch.settings.max_cholesky_size(
            self.max_cholesky_size
        ):
            x_cand_torch = torch.tensor(x_cand).to(device=device, dtype=dtype)
            y_cand = (
                gp.likelihood(gp(x_cand_torch))
                .sample(torch.Size([self.batch_size]))
                .t()
                .cpu()
                .detach()
                .numpy()
            )

        # Remove the torch variables
        del x_torch, y_torch, x_cand_torch, gp

        # De-standardize the sampled values
        y_cand = mu + sigma * y_cand

        return x_cand, y_cand, self.hypers

    def update(self, completed_trials):

        # Filter only the trials for this trust region
        trials_tr = []
        x = np.zeros((0, self.num_dims))
        y = np.zeros((0, 1))

        for t in completed_trials:
            if t.trust_region.id == self.id:
                trials_tr.append(t)
                x = np.vstack((x, t.x))
                y = np.append(y, t.y)

        if len(trials_tr) > 0:
            self.hypers = {}
            self.adjust_length(y)

            # Now append the trials
            self.trials = np.append(self.trials, trials_tr)

        # Check if trust region must be restarted
        if self.length < self.length_min:  # Restart trust region if converged
            best_trial = self.get_best_trial()
            print(
                f"{len(self.trials)}) TR-{self.id}_{self.num_restarts} converged to: : {best_trial.y:.4}"
            )
            sys.stdout.flush()
            self.restart()

    def adjust_length(self, y):
        best_trial = self.get_best_trial()
        best_value = np.inf if best_trial is None else best_trial.y  # Target value
        # print(f"TR-{self.id} - best: {best_value}, y.min(): {y.min()}")

        if y.min() < best_value - 1e-3 * math.fabs(best_value):
            self.succcount += 1
            self.failcount = 0
        else:
            self.succcount = 0
            self.failcount += len(y)  # NOTE: Add size of the batch for this TR

        if self.succcount == self.success_tolerance:  # Expand trust region
            self.length = min([2.0 * self.length, self.length_max])
            print(
                f"Expanded TR-{self.id}_{self.num_restarts}, new length: {self.length}"
            )
            self.succcount = 0
        elif (
            self.failcount >= self.failure_tolerance
        ):  # Shrink trust region (we may have exceeded the failtol)
            self.length /= 2.0
            print(f"Shrunk TR-{self.id}_{self.num_restarts}, new length: {self.length}")
            self.failcount = 0
        pass

    def name(self):
        return str(self.id) + "_" + str(self.num_restarts)

    def get_best_trial(self):
        best_trial = None
        best_y = np.inf
        for t in self.trials:
            if t.status == "completed":
                if t.y < best_y:
                    best_y = t.y
                    best_trial = t

        return best_trial

    def get_num_trials(self):
        return len(self.trials)

    def __str__(self):
        return f"TR(name={self.name()}, trials={len(self.trials)})"

    def __repr__(self):
        return f"TR(name={self.name()}, trials={len(self.trials)})"

class Trial:
    def __init__(self, trial_no, trust_region, type="ts") -> None:
        self.trial_no = trial_no
        self.trust_region = trust_region
        self.trust_region_length = trust_region.length
        self.trust_region_weights = np.copy(trust_region.weights)
        self.trust_region_lower_bounds = np.copy(trust_region.lb)
        self.trust_region_upper_bounds = np.copy(trust_region.ub)
        self.trust_region_center_x = np.copy(trust_region.center_x)
        self.trust_region_center_y = trust_region.center_y
        self.type = type
        self.batch_no = -1
        self.status = "open"
        self.x = np.zeros((0, 1))
        self.y = np.inf

    def set_x(self, x):
        self.x = x
    
    def set_batch_no(self, batch_no):
        self.batch_no = batch_no

    def complete(self, y):
        self.y = y
        self.status = "completed"

    def data(self):
        return {
            "trial_no": self.trial_no,
            "batch_no": self.batch_no,
            "trust_region_name": self.trust_region.name(),
            "trust_region_length": self.trust_region_length,
            "x": self.x.tolist(),
            "y": self.y,
            "trust_region_weights": self.trust_region_weights.tolist(),
            "trust_region_lower_bounds": self.trust_region_lower_bounds.tolist(),
            "trust_region_upper_bounds": self.trust_region_upper_bounds.tolist(),
            "trust_region_center_x": self.trust_region_center_x.tolist(),
            "trust_region_center_y": self.trust_region_center_y,
            "type": self.type,
            "status": self.status,
        }

    def __str__(self):
        return f"Trial(trial_no={self.trial_no}, trust_region={self.trust_region}, x={self.x}, y={self.y}, type={self.type}, status={self.status}"

    def __repr__(self):
        return f"Trial(trial_no={self.trial_no}, trust_region={self.trust_region}, x={self.x}, y={self.y}, type={self.type}, status={self.status}"
